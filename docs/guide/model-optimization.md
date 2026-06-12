# Model Optimization & Pareto Frontiers

AnamDB is built to handle multiple neural model candidate configurations for the same SQL query operators, balancing latency, accuracy, and compute costs.

---

## The Optimization Problem

When evaluating neural queries, traditional database kernels look at access paths (indexes, scans). AnamDB extends this to look at **Model Access Paths**:
1. **Fast, low-accuracy models:** (e.g., highly quantized ONNX classifiers run on CPU in 0.5ms with 75% accuracy).
2. **Slow, high-accuracy models:** (e.g., full precision ONNX classifiers running on CUDA/Metal in 5.0ms with 95% accuracy).

---

## Pareto Frontier Selection

AnamDB's multi-objective optimizer selects candidate execution plans that are not dominated by any other options. A plan dominates another if it is better in at least one objective and equal or better in all others.

During execution, the optimizer:
1. Gathers all registered Function-as-Operator (FAO) targets for a query signature.
2. Scales estimated execution times based on active [device pools](../cli/repl/devices).
3. Computes the Pareto frontier curve.
4. Picks the execution target matching the user session's configuration parameters.
