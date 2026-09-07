"""Export the same trained weights to multiple on-device runtimes (Phase 9).

Targets: GGUF (llama.cpp), ONNX (ONNX Runtime), ExecuTorch. Shared Q4_0-style quantizer +
a parity test (eval.harness.parity_check) ensure outputs match the PyTorch reference.
"""
