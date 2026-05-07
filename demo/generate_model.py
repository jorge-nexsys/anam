#!/usr/bin/env python3
"""Generate ONNX model variants for Pareto optimization demos.

Creates two models:
  1. fraud_detector.onnx     — accurate (full 3-feature, sigmoid, ~0.95 acc)
  2. fraud_detector_fast.onnx — fast (single feature, linear, ~0.75 acc)

The fast model only uses fraud_prob directly, simulating a lightweight
model that trades accuracy for latency.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import os


def create_accurate_model():
    """Full 3-feature fraud detector with sigmoid (high accuracy, higher latency)."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    weights = np.array([[0.00001], [3.0], [0.1]], dtype=np.float32)
    bias = np.array([-2.0], dtype=np.float32)

    W = numpy_helper.from_array(weights, name="W")
    B = numpy_helper.from_array(bias, name="B")

    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"])
    sigmoid = helper.make_node("Sigmoid", ["add_out"], ["output"])

    graph = helper.make_graph([matmul, add, sigmoid], "fraud_detector_accurate", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def create_fast_model():
    """Single-feature linear model (low latency, lower accuracy).
    
    Simply passes fraud_prob through with a scale+bias:
      output = scale * input[:, 1] + bias
    But since we need it as [batch, 3] input for compatibility,
    we just use a simpler weight vector that only looks at col[1].
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    # Only the fraud_prob column matters; ignore amount and hour.
    weights = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)

    W = numpy_helper.from_array(weights, name="W")
    B = numpy_helper.from_array(bias, name="B")

    # No sigmoid — just linear pass-through (faster).
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "B"], ["output"])

    graph = helper.make_graph([matmul, add], "fraud_detector_fast", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    os.makedirs("demo/models", exist_ok=True)

    # Accurate model
    accurate = create_accurate_model()
    onnx.save(accurate, "demo/models/fraud_detector.onnx")
    print(f"✓ fraud_detector.onnx       ({os.path.getsize('demo/models/fraud_detector.onnx')} bytes) — accurate, slower")

    # Fast model
    fast = create_fast_model()
    onnx.save(fast, "demo/models/fraud_detector_fast.onnx")
    print(f"✓ fraud_detector_fast.onnx  ({os.path.getsize('demo/models/fraud_detector_fast.onnx')} bytes) — fast, less accurate")
