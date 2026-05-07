//! Heterogeneous hardware dispatcher.
//!
//! Manages a pool of available compute devices (CPU, GPU via CUDA/Metal, NPU)
//! and schedules fine-grained execution jobs across them.

use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
#[cfg(any(target_os = "macos", feature = "cuda"))]
use tracing::warn;
use tracing::{debug, info};

use crate::core::error::{AnamError, Result};

/// Type of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// General-purpose CPU core(s).
    Cpu,
    /// NVIDIA GPU (CUDA).
    CudaGpu,
    /// Apple GPU (Metal).
    MetalGpu,
    /// Neural Processing Unit (CoreML / other).
    Npu,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::CudaGpu => write!(f, "CUDA GPU"),
            DeviceType::MetalGpu => write!(f, "Metal GPU"),
            DeviceType::Npu => write!(f, "NPU"),
        }
    }
}

/// A single compute device slot.
#[derive(Debug, Clone)]
pub struct DeviceSlot {
    /// Unique slot identifier.
    pub id: usize,
    /// Type of device.
    pub device_type: DeviceType,
    /// Human-readable name (e.g. "Apple M3 Pro GPU").
    pub name: String,
    /// Available memory in bytes (0 for CPU slots).
    pub available_memory_bytes: u64,
    /// Total memory in bytes.
    pub total_memory_bytes: u64,
    /// Speed multiplier relative to a baseline CPU (1.0 = CPU baseline).
    pub speed_factor: f64,
}

/// A fine-grained execution job to be dispatched.
#[derive(Debug)]
pub struct ExecutionJob {
    /// Unique job ID.
    pub job_id: String,
    /// Preferred device type (if any).
    pub preferred_device: Option<DeviceType>,
    /// Estimated compute time on CPU (milliseconds).
    pub est_cpu_time_ms: f64,
    /// Estimated memory requirement (bytes).
    pub est_memory_bytes: u64,
}

/// Assignment of a job to a specific device slot.
#[derive(Debug)]
pub struct JobAssignment {
    /// The job.
    pub job: ExecutionJob,
    /// Assigned device slot.
    pub slot: DeviceSlot,
    /// Estimated time on the assigned device.
    pub est_time_ms: f64,
}

/// Pool of available compute devices with load-balancing dispatch.
pub struct DevicePool {
    /// Available device slots.
    slots: Vec<DeviceSlot>,
    /// Round-robin counter for load balancing.
    _next_slot: AtomicUsize,
    /// Current load per slot (number of active jobs).
    slot_loads: Vec<AtomicUsize>,
}

impl std::fmt::Debug for DevicePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePool")
            .field("slots", &self.slots)
            .finish()
    }
}

impl DevicePool {
    /// Create a CPU-only pool (uses all available cores).
    pub fn cpu_only() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let slots: Vec<DeviceSlot> = (0..num_cpus)
            .map(|i| DeviceSlot {
                id: i,
                device_type: DeviceType::Cpu,
                name: format!("CPU-{i}"),
                available_memory_bytes: 0,
                total_memory_bytes: 0,
                speed_factor: 1.0,
            })
            .collect();

        let slot_loads: Vec<AtomicUsize> = (0..slots.len()).map(|_| AtomicUsize::new(0)).collect();

        info!(cpu_slots = num_cpus, "initialized CPU-only device pool");

        Self {
            slots,
            _next_slot: AtomicUsize::new(0),
            slot_loads,
        }
    }

    /// Auto-detect available hardware and create a heterogeneous pool.
    pub async fn detect_hardware() -> Result<Self> {
        let mut slots = Vec::new();
        let mut slot_id = 0;

        // Always include CPU slots.
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        for i in 0..num_cpus {
            slots.push(DeviceSlot {
                id: slot_id,
                device_type: DeviceType::Cpu,
                name: format!("CPU-{i}"),
                available_memory_bytes: 0,
                total_memory_bytes: 0,
                speed_factor: 1.0,
            });
            slot_id += 1;
        }

        // ── Metal GPU detection (macOS) ────────────────────────────────
        #[cfg(target_os = "macos")]
        {
            match Self::detect_metal_devices(&mut slot_id) {
                Ok(metal_slots) => {
                    info!(count = metal_slots.len(), "detected Metal GPU devices");
                    slots.extend(metal_slots);
                }
                Err(e) => {
                    warn!("Metal detection failed: {e}");
                }
            }
        }

        // ── CUDA GPU detection ─────────────────────────────────────────
        #[cfg(feature = "cuda")]
        {
            match Self::detect_cuda_devices(&mut slot_id) {
                Ok(cuda_slots) => {
                    info!(count = cuda_slots.len(), "detected CUDA GPU devices");
                    slots.extend(cuda_slots);
                }
                Err(e) => {
                    warn!("CUDA detection failed: {e}");
                }
            }
        }

        // ── nvidia-smi fallback (no CUDA runtime needed) ──────────────
        #[cfg(not(feature = "cuda"))]
        {
            match Self::detect_nvidia_smi_devices(&mut slot_id) {
                Ok(nvidia_slots) => {
                    if !nvidia_slots.is_empty() {
                        info!(
                            count = nvidia_slots.len(),
                            "detected NVIDIA GPUs via nvidia-smi"
                        );
                        slots.extend(nvidia_slots);
                    }
                }
                Err(e) => {
                    debug!("nvidia-smi not available: {e}");
                }
            }
        }

        let slot_loads: Vec<AtomicUsize> = (0..slots.len()).map(|_| AtomicUsize::new(0)).collect();

        info!(
            total_slots = slots.len(),
            gpu_slots = slots
                .iter()
                .filter(|s| matches!(s.device_type, DeviceType::CudaGpu | DeviceType::MetalGpu))
                .count(),
            "device pool initialized"
        );

        Ok(Self {
            slots,
            _next_slot: AtomicUsize::new(0),
            slot_loads,
        })
    }

    /// Detect Metal GPU devices on macOS.
    #[cfg(target_os = "macos")]
    fn detect_metal_devices(slot_id: &mut usize) -> Result<Vec<DeviceSlot>> {
        use std::process::Command;

        let mut devices = Vec::new();

        // Use system_profiler to detect GPU info.
        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .map_err(|e| AnamError::Dispatch(format!("failed to run system_profiler: {e}")))?;

        if output.status.success() {
            let json: serde_json::Value =
                serde_json::from_slice(&output.stdout).unwrap_or(serde_json::Value::Null);

            // Extract GPU names from system profiler output.
            if let Some(displays) = json.get("SPDisplaysDataType").and_then(|v| v.as_array()) {
                for display in displays {
                    let name = display
                        .get("sppci_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Apple GPU");

                    // Estimate VRAM from chipset info.
                    let vram_str = display
                        .get("spdisplays_vram_shared")
                        .or_else(|| display.get("spdisplays_vram"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("0");

                    let vram_bytes = parse_memory_string(vram_str);

                    devices.push(DeviceSlot {
                        id: *slot_id,
                        device_type: DeviceType::MetalGpu,
                        name: format!("Metal: {name}"),
                        available_memory_bytes: vram_bytes,
                        total_memory_bytes: vram_bytes,
                        // Apple Silicon GPUs are roughly 5-20x faster than CPU for ML.
                        speed_factor: 10.0,
                    });
                    *slot_id += 1;
                }
            }
        }

        // Fallback: if system_profiler didn't find anything, add a generic Metal device.
        if devices.is_empty() {
            devices.push(DeviceSlot {
                id: *slot_id,
                device_type: DeviceType::MetalGpu,
                name: "Metal: Apple GPU (generic)".into(),
                available_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB estimate
                total_memory_bytes: 8 * 1024 * 1024 * 1024,
                speed_factor: 10.0,
            });
            *slot_id += 1;
        }

        Ok(devices)
    }

    /// Detect CUDA GPU devices.
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices(slot_id: &mut usize) -> Result<Vec<DeviceSlot>> {
        use cudarc::driver::CudaDevice;

        let mut devices = Vec::new();

        // Enumerate CUDA devices.
        let device_count = CudaDevice::count()
            .map_err(|e| AnamError::Dispatch(format!("CUDA device enumeration failed: {e}")))?;

        for ordinal in 0..device_count {
            let dev = CudaDevice::new(ordinal).map_err(|e| {
                AnamError::Dispatch(format!("CUDA device {ordinal} init failed: {e}"))
            })?;

            let (free, total) = dev
                .mem_get_info()
                .map_err(|e| AnamError::Dispatch(format!("CUDA mem query failed: {e}")))?;

            let name = format!("CUDA:{ordinal}");

            devices.push(DeviceSlot {
                id: *slot_id,
                device_type: DeviceType::CudaGpu,
                name,
                available_memory_bytes: free as u64,
                total_memory_bytes: total as u64,
                speed_factor: 50.0, // NVIDIA GPUs are ~50x for dense compute.
            });
            *slot_id += 1;
        }

        Ok(devices)
    }

    /// Detect NVIDIA GPUs via `nvidia-smi` CLI (no CUDA runtime needed).
    ///
    /// This is a fallback for systems where the `cuda` cargo feature is not
    /// enabled but an NVIDIA GPU is present.
    #[cfg(not(feature = "cuda"))]
    fn detect_nvidia_smi_devices(slot_id: &mut usize) -> Result<Vec<DeviceSlot>> {
        use std::process::Command;

        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| AnamError::Dispatch(format!("nvidia-smi not found: {e}")))?;

        if !output.status.success() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut devices = Vec::new();

        for line in stdout.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let name = parts[0].to_string();
                let free_mib: u64 = parts[1].parse().unwrap_or(0);
                let total_mib: u64 = parts[2].parse().unwrap_or(0);

                devices.push(DeviceSlot {
                    id: *slot_id,
                    device_type: DeviceType::CudaGpu,
                    name: format!("CUDA: {name}"),
                    available_memory_bytes: free_mib * 1024 * 1024,
                    total_memory_bytes: total_mib * 1024 * 1024,
                    speed_factor: 50.0,
                });
                *slot_id += 1;
            }
        }

        Ok(devices)
    }

    // ── Dispatch ───────────────────────────────────────────────────────

    /// Dispatch a job to the best available device slot.
    pub fn dispatch(&self, job: ExecutionJob) -> Result<JobAssignment> {
        // Strategy: prefer the device type requested by the job, then pick
        // the slot with the lowest current load.
        let candidate_slots: Vec<&DeviceSlot> = if let Some(pref) = job.preferred_device {
            let preferred: Vec<_> = self
                .slots
                .iter()
                .filter(|s| s.device_type == pref)
                .collect();
            if preferred.is_empty() {
                // Fallback to any slot.
                self.slots.iter().collect()
            } else {
                preferred
            }
        } else {
            self.slots.iter().collect()
        };

        // Pick the slot with the lowest load.
        let best_slot = candidate_slots
            .iter()
            .min_by_key(|slot| self.slot_loads[slot.id].load(Ordering::Relaxed))
            .ok_or_else(|| AnamError::Dispatch("no available device slots".into()))?;

        // Increment load.
        self.slot_loads[best_slot.id].fetch_add(1, Ordering::Relaxed);

        let est_time_ms = job.est_cpu_time_ms / best_slot.speed_factor;

        debug!(
            job_id = %job.job_id,
            device = %best_slot.name,
            est_time_ms = est_time_ms,
            "dispatched job"
        );

        Ok(JobAssignment {
            job,
            slot: (*best_slot).clone(),
            est_time_ms,
        })
    }

    /// Mark a job as completed, releasing the slot load.
    pub fn complete_job(&self, assignment: &JobAssignment) {
        self.slot_loads[assignment.slot.id].fetch_sub(1, Ordering::Relaxed);
    }

    /// Get the speed multiplier of the fastest available device.
    pub fn speed_multiplier(&self) -> f64 {
        self.slots
            .iter()
            .map(|s| s.speed_factor)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0)
    }

    /// List all available slots.
    pub fn list_slots(&self) -> &[DeviceSlot] {
        &self.slots
    }

    /// Get a summary of the pool for display.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        for slot in &self.slots {
            let load = self.slot_loads[slot.id].load(Ordering::Relaxed);
            let mem = if slot.total_memory_bytes > 0 {
                format!(
                    " [{}/{} MB]",
                    slot.available_memory_bytes / (1024 * 1024),
                    slot.total_memory_bytes / (1024 * 1024)
                )
            } else {
                String::new()
            };
            lines.push(format!(
                "  [{:>2}] {} ({}x){} — load: {}",
                slot.id, slot.name, slot.speed_factor, mem, load
            ));
        }
        lines.join("\n")
    }
}

/// Parse a memory string like "8 GB" or "16384 MB" into bytes.
#[cfg(target_os = "macos")]
fn parse_memory_string(s: &str) -> u64 {
    let s = s.trim();
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() >= 2 {
        let value: f64 = parts[0].parse().unwrap_or(0.0);
        match parts[1].to_uppercase().as_str() {
            "TB" => (value * 1024.0 * 1024.0 * 1024.0 * 1024.0) as u64,
            "GB" => (value * 1024.0 * 1024.0 * 1024.0) as u64,
            "MB" => (value * 1024.0 * 1024.0) as u64,
            "KB" => (value * 1024.0) as u64,
            _ => value as u64,
        }
    } else {
        s.parse().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_pool() {
        let pool = DevicePool::cpu_only();
        assert!(!pool.slots.is_empty());
        assert!(pool.slots.iter().all(|s| s.device_type == DeviceType::Cpu));
    }

    #[test]
    fn dispatch_round_robin() {
        let pool = DevicePool::cpu_only();

        let job1 = ExecutionJob {
            job_id: "j1".into(),
            preferred_device: None,
            est_cpu_time_ms: 10.0,
            est_memory_bytes: 0,
        };
        let assignment1 = pool.dispatch(job1).unwrap();
        assert_eq!(assignment1.slot.device_type, DeviceType::Cpu);

        // Second job should go to a different (or same with lowest load) slot.
        let job2 = ExecutionJob {
            job_id: "j2".into(),
            preferred_device: None,
            est_cpu_time_ms: 10.0,
            est_memory_bytes: 0,
        };
        let _assignment2 = pool.dispatch(job2).unwrap();

        // Complete the first job.
        pool.complete_job(&assignment1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn parse_memory() {
        assert_eq!(parse_memory_string("8 GB"), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_string("16384 MB"), 16384 * 1024 * 1024);
    }
}
