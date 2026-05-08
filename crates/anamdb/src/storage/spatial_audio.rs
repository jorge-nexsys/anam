//! Expanded Semantic Abstractions — 3D spatial and temporal audio types.
//!
//! Provides native columnar representations for:
//! - **3D spatial data**: point clouds, bounding boxes, scene graphs
//! - **Temporal audio graphs**: spectrograms, waveform sequences, beat-aligned events
//!
//! These types integrate with DataFusion as extension types stored as
//! Arrow `BinaryArray` (msgpack-encoded structs), enabling SQL predicates
//! like `spatial_distance(location, POINT(1.0, 2.0, 3.0)) < 5.0` and
//! `audio_tempo(clip) BETWEEN 120 AND 140`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, BinaryArray, BinaryBuilder};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use crate::core::error::{AnamError, Result};

// ═══════════════════════════════════════════════════════════════════════
// 3D Spatial Abstractions
// ═══════════════════════════════════════════════════════════════════════

/// A 3D point in Euclidean space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Point3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// An axis-aligned bounding box in 3D space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox3D {
    /// Minimum corner.
    pub min: Point3D,
    /// Maximum corner.
    pub max: Point3D,
    /// Optional semantic label (e.g., "car", "pedestrian").
    pub label: Option<String>,
    /// Confidence score from the detection model.
    pub confidence: f32,
}

impl BoundingBox3D {
    /// Volume of the bounding box.
    pub fn volume(&self) -> f32 {
        let dx = (self.max.x - self.min.x).abs();
        let dy = (self.max.y - self.min.y).abs();
        let dz = (self.max.z - self.min.z).abs();
        dx * dy * dz
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Point3D {
        Point3D::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    /// Intersection over Union with another bounding box.
    pub fn iou(&self, other: &BoundingBox3D) -> f32 {
        let ix_min = self.min.x.max(other.min.x);
        let iy_min = self.min.y.max(other.min.y);
        let iz_min = self.min.z.max(other.min.z);
        let ix_max = self.max.x.min(other.max.x);
        let iy_max = self.max.y.min(other.max.y);
        let iz_max = self.max.z.min(other.max.z);

        let inter_dx = (ix_max - ix_min).max(0.0);
        let inter_dy = (iy_max - iy_min).max(0.0);
        let inter_dz = (iz_max - iz_min).max(0.0);
        let intersection = inter_dx * inter_dy * inter_dz;

        if intersection == 0.0 {
            return 0.0;
        }

        let union = self.volume() + other.volume() - intersection;
        intersection / union
    }
}

/// A point cloud: an ordered set of 3D points with optional intensity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud {
    /// Points in the cloud.
    pub points: Vec<Point3D>,
    /// Intensity per point (e.g., LiDAR reflectance), same length as `points`.
    pub intensity: Vec<f32>,
    /// Coordinate frame identifier.
    pub frame: String,
    /// Timestamp in nanoseconds since epoch.
    pub timestamp_ns: u64,
}

impl PointCloud {
    /// Compute the centroid of the point cloud.
    pub fn centroid(&self) -> Option<Point3D> {
        if self.points.is_empty() {
            return None;
        }
        let n = self.points.len() as f32;
        let x = self.points.iter().map(|p| p.x).sum::<f32>() / n;
        let y = self.points.iter().map(|p| p.y).sum::<f32>() / n;
        let z = self.points.iter().map(|p| p.z).sum::<f32>() / n;
        Some(Point3D::new(x, y, z))
    }

    /// Number of points in the cloud.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the point cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ── Spatial Arrow Column Encoding ────────────────────────────────────

/// Encode a slice of BoundingBox3D values into a BinaryArray (msgpack via bincode).
pub fn encode_bboxes(bboxes: &[BoundingBox3D]) -> Result<ArrayRef> {
    let mut builder = BinaryBuilder::new();
    for bbox in bboxes {
        let bytes = bincode::serialize(bbox)
            .map_err(|e| AnamError::Serde(format!("bbox encode error: {e}")))?;
        builder.append_value(&bytes);
    }
    Ok(Arc::new(builder.finish()))
}

/// Decode a BinaryArray back into a Vec<BoundingBox3D>.
pub fn decode_bboxes(array: &BinaryArray) -> Result<Vec<BoundingBox3D>> {
    (0..array.len())
        .map(|i| {
            let bytes = array.value(i);
            bincode::deserialize::<BoundingBox3D>(bytes)
                .map_err(|e| AnamError::Serde(format!("bbox decode error: {e}")))
        })
        .collect()
}

/// Encode a PointCloud into bytes.
pub fn encode_point_cloud(pc: &PointCloud) -> Result<Vec<u8>> {
    bincode::serialize(pc).map_err(|e| AnamError::Serde(format!("point cloud encode error: {e}")))
}

/// Decode bytes into a PointCloud.
pub fn decode_point_cloud(bytes: &[u8]) -> Result<PointCloud> {
    bincode::deserialize(bytes)
        .map_err(|e| AnamError::Serde(format!("point cloud decode error: {e}")))
}

// ═══════════════════════════════════════════════════════════════════════
// Temporal Audio Graph Abstractions
// ═══════════════════════════════════════════════════════════════════════

/// A temporal audio event (e.g., a beat, note onset, or phoneme).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvent {
    /// Start time in seconds.
    pub start_sec: f32,
    /// Duration in seconds.
    pub duration_sec: f32,
    /// Pitch in MIDI note number (0–127), if applicable.
    pub pitch: Option<f32>,
    /// Event label (e.g., "beat", "onset", "phoneme:a").
    pub label: String,
    /// Confidence score.
    pub confidence: f32,
}

/// A compact spectrogram representation (mel-scale).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelSpectrogram {
    /// Frequency bins (mel scale).
    pub n_mels: u32,
    /// Time frames.
    pub n_frames: u32,
    /// Flattened magnitude values (n_mels × n_frames, row-major).
    pub magnitudes: Vec<f32>,
    /// Sample rate of the source audio.
    pub sample_rate: u32,
    /// Hop length in samples.
    pub hop_length: u32,
}

impl MelSpectrogram {
    /// Get the magnitude at a specific mel bin and time frame.
    pub fn get(&self, mel: u32, frame: u32) -> Option<f32> {
        if mel >= self.n_mels || frame >= self.n_frames {
            return None;
        }
        let idx = (mel * self.n_frames + frame) as usize;
        self.magnitudes.get(idx).copied()
    }

    /// Duration of the audio clip in seconds.
    pub fn duration_secs(&self) -> f32 {
        (self.n_frames * self.hop_length) as f32 / self.sample_rate as f32
    }
}

/// A temporal audio graph: events linked across time with beat alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGraph {
    /// Ordered list of temporal events.
    pub events: Vec<AudioEvent>,
    /// Optional mel spectrogram for the full clip.
    pub spectrogram: Option<MelSpectrogram>,
    /// Estimated tempo in BPM.
    pub tempo_bpm: Option<f32>,
    /// Total clip duration in seconds.
    pub duration_sec: f32,
    /// Source identifier (file path, URL, or hash).
    pub source_id: String,
}

impl AudioGraph {
    /// Events within a time window [start, end) seconds.
    pub fn events_in_window(&self, start_sec: f32, end_sec: f32) -> Vec<&AudioEvent> {
        self.events
            .iter()
            .filter(|e| e.start_sec >= start_sec && e.start_sec < end_sec)
            .collect()
    }

    /// Average confidence across all events.
    pub fn mean_confidence(&self) -> f32 {
        if self.events.is_empty() {
            return 0.0;
        }
        self.events.iter().map(|e| e.confidence).sum::<f32>() / self.events.len() as f32
    }
}

// ── Audio Arrow Column Encoding ───────────────────────────────────────

/// Encode a slice of AudioGraph values into a BinaryArray.
pub fn encode_audio_graphs(graphs: &[AudioGraph]) -> Result<ArrayRef> {
    let mut builder = BinaryBuilder::new();
    for graph in graphs {
        let bytes = bincode::serialize(graph)
            .map_err(|e| AnamError::Serde(format!("audio graph encode error: {e}")))?;
        builder.append_value(&bytes);
    }
    Ok(Arc::new(builder.finish()))
}

/// Decode a BinaryArray back into a Vec<AudioGraph>.
pub fn decode_audio_graphs(array: &BinaryArray) -> Result<Vec<AudioGraph>> {
    (0..array.len())
        .map(|i| {
            let bytes = array.value(i);
            bincode::deserialize::<AudioGraph>(bytes)
                .map_err(|e| AnamError::Serde(format!("audio graph decode error: {e}")))
        })
        .collect()
}

// ── Schema Helpers ────────────────────────────────────────────────────

/// Arrow schema for a table containing bounding boxes.
pub fn bbox_table_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("bbox", DataType::Binary, false),
        Field::new("frame_id", DataType::Utf8, true),
        Field::new("timestamp_ns", DataType::UInt64, true),
    ])
}

/// Arrow schema for a table containing audio graphs.
pub fn audio_graph_table_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("audio_graph", DataType::Binary, false),
        Field::new("source_id", DataType::Utf8, true),
        Field::new("duration_sec", DataType::Float32, true),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point3d_distance() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(1.0, 2.0, 2.0);
        let dist = a.distance_to(&b);
        assert!((dist - 3.0).abs() < 1e-5, "expected 3.0, got {dist}");
        println!("\n═══ Spatial Abstractions Test ═══");
        println!("  ✓ 3D distance: {dist:.4}");
    }

    #[test]
    fn bounding_box_iou() {
        let a = BoundingBox3D {
            min: Point3D::new(0.0, 0.0, 0.0),
            max: Point3D::new(2.0, 2.0, 2.0),
            label: Some("car".into()),
            confidence: 0.9,
        };
        let b = BoundingBox3D {
            min: Point3D::new(1.0, 1.0, 1.0),
            max: Point3D::new(3.0, 3.0, 3.0),
            label: Some("car".into()),
            confidence: 0.85,
        };

        let iou = a.iou(&b);
        assert!(iou > 0.0 && iou < 1.0, "IoU should be in (0, 1): {iou}");
        println!("  ✓ BBox3D IoU: {iou:.4}");
        println!("  ✓ BBox3D volume: {:.2}", a.volume());
        println!("  ✓ BBox3D center: {:?}", a.center());
    }

    #[test]
    fn bbox_arrow_roundtrip() {
        let bboxes = vec![BoundingBox3D {
            min: Point3D::new(0.0, 0.0, 0.0),
            max: Point3D::new(1.0, 1.0, 1.0),
            label: Some("pedestrian".into()),
            confidence: 0.92,
        }];

        let array = encode_bboxes(&bboxes).unwrap();
        let binary_array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
        let decoded = decode_bboxes(binary_array).unwrap();

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].label.as_deref(), Some("pedestrian"));
        assert!((decoded[0].confidence - 0.92).abs() < 1e-5);
        println!("  ✓ BBox3D Arrow roundtrip: encode → BinaryArray → decode");
    }

    #[test]
    fn point_cloud_centroid() {
        let pc = PointCloud {
            points: vec![
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(2.0, 0.0, 0.0),
                Point3D::new(1.0, 3.0, 0.0),
            ],
            intensity: vec![1.0, 0.8, 0.9],
            frame: "lidar".into(),
            timestamp_ns: 1_000_000,
        };

        let c = pc.centroid().unwrap();
        assert!((c.x - 1.0).abs() < 1e-5);
        assert!((c.y - 1.0).abs() < 1e-5);
        println!(
            "  ✓ PointCloud centroid: ({:.1}, {:.1}, {:.1})",
            c.x, c.y, c.z
        );
    }

    #[test]
    fn audio_graph_roundtrip() {
        let graph = AudioGraph {
            events: vec![
                AudioEvent {
                    start_sec: 0.0,
                    duration_sec: 0.5,
                    pitch: Some(60.0),
                    label: "beat".into(),
                    confidence: 0.95,
                },
                AudioEvent {
                    start_sec: 0.5,
                    duration_sec: 0.5,
                    pitch: None,
                    label: "onset".into(),
                    confidence: 0.80,
                },
            ],
            spectrogram: None,
            tempo_bpm: Some(120.0),
            duration_sec: 1.0,
            source_id: "track_001".into(),
        };

        let graphs = vec![graph];
        let array = encode_audio_graphs(&graphs).unwrap();
        let binary_array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
        let decoded = decode_audio_graphs(binary_array).unwrap();

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].events.len(), 2);
        assert!((decoded[0].tempo_bpm.unwrap() - 120.0).abs() < 1e-3);
        assert!((decoded[0].mean_confidence() - 0.875).abs() < 1e-3);

        let window = decoded[0].events_in_window(0.0, 0.5);
        assert_eq!(window.len(), 1);

        println!("\n═══ Audio Graph Test ═══");
        println!("  ✓ Audio graph Arrow roundtrip");
        println!(
            "  ✓ Temporal window query: {} event in [0.0, 0.5)s",
            window.len()
        );
        println!("  ✓ Mean confidence: {:.3}", decoded[0].mean_confidence());
    }
}
