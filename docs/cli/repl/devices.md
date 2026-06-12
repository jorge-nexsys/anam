# REPL: Compute Device Pool

Monitor and evaluate active compute hardware acceleration bindings.

---

## `.devices`
Outputs active hardware compute device metrics (like available processors or GPU runtimes).

* **Syntax**: `.devices`
* **Output Detail**:
  * Shows CPU execution threads.
  * Lists any active graphic APIs (Metal on macOS, CUDA on Nvidia platforms) and their estimated computation performance multiplier metrics.

---

## Example

```sql
anam> .devices
═══ Device Pool ═══
[ 0–7] CPU-0..CPU-7 (1x) — idle
[   8] Metal: Apple M2 (10x) — idle
```
