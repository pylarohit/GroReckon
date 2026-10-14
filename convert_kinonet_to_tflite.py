"""
================================================================================
KinoNet-R2 PyTorch → TFLite Converter
================================================================================

Run this script ONCE (on your laptop/Kaggle) to convert the trained PyTorch
model kinonet_r2_best.pth into kinonet_r2.tflite for Android deployment.

Output:  kinonet_r2.tflite  (copy to app/src/main/assets/)

Usage:
    python convert_kinonet_to_tflite.py

Requirements:
    pip install torch onnx onnxruntime tensorflow tf2onnx

Steps:
    1. PyTorch  → ONNX  (torch.onnx.export)
    2. ONNX     → TFLite (tf2onnx + TFLite converter)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# ---- KinoNet-R2 definition (must match georeckon_pipeline.py exactly) ----
class KinoNetR2(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_dim // 2)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1   = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out= nn.Linear(hidden_dim // 2, 2)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = h.mean(dim=2)                        # Global average pool
        h = F.relu(self.fc1(h))
        out = self.fc_out(h)                      # (batch, 2) → [μ, log_σ²]
        out[:, 1] = torch.clamp(out[:, 1], -10.0, 10.0)
        return out                                # [batch, 2]


def convert_to_tflite(pth_path: str, out_path: str = "kinonet_r2.tflite"):
    print(f"[1/4] Loading PyTorch model from: {pth_path}")
    model = KinoNetR2()
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ---- Step 1: PyTorch → ONNX ----
    print("[2/4] Exporting to ONNX...")
    dummy_input = torch.randn(1, 6, 20)  # [batch=1, channels=6, window=20]
    onnx_path = pth_path.replace(".pth", ".onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["imu_window"],
        output_names=["speed_logvar"],
        dynamic_axes={
            "imu_window":   {0: "batch_size"},
            "speed_logvar": {0: "batch_size"},
        },
    )
    print(f"    ONNX saved: {onnx_path}")

    # ---- Step 2: ONNX → TensorFlow SavedModel ----
    print("[3/4] Converting ONNX → TFLite...")
    try:
        import subprocess
        saved_model_dir = pth_path.replace(".pth", "_saved_model")
        result = subprocess.run([
            "python", "-m", "tf2onnx.convert",
            "--onnx", onnx_path,
            "--output", saved_model_dir,
            "--opset", "17",
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        # TFLite conversion
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]      # float16 quantization
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        with open(out_path, "wb") as f:
            f.write(tflite_model)
        print(f"    TFLite saved: {out_path}")
        print(f"    Model size: {os.path.getsize(out_path) / 1024:.1f} KB")

    except ImportError:
        print("[WARN] tf2onnx or tensorflow not installed.")
        print("       Install: pip install tensorflow tf2onnx")
        print("       Falling back to ONNX Runtime verification only.")
        _verify_onnx(onnx_path, dummy_input.numpy())
        return

    # ---- Step 4: Verify TFLite output ----
    print("[4/4] Verifying TFLite model...")
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=out_path)
        interp.allocate_tensors()

        input_details  = interp.get_input_details()
        output_details = interp.get_output_details()

        test_input = np.random.randn(1, 6, 20).astype(np.float32)
        interp.set_tensor(input_details[0]["index"], test_input)
        interp.invoke()
        output = interp.get_tensor(output_details[0]["index"])
        print(f"    Test output: μ={output[0,0]:.4f}  log_σ²={output[0,1]:.4f}")
        print("[DONE] TFLite conversion successful!")
        print(f"\nNext step: copy {out_path}")
        print("       to:  app/src/main/assets/kinonet_r2.tflite")
    except Exception as e:
        print(f"[WARN] TFLite verification failed: {e}")


def _verify_onnx(onnx_path: str, input_np: np.ndarray):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        out = sess.run(None, {"imu_window": input_np})
        print(f"    ONNX output: {out[0]}")
        print("[DONE] ONNX model verified (TFLite step skipped).")
    except ImportError:
        print("[WARN] onnxruntime not installed. Skipping verification.")


if __name__ == "__main__":
    pth = "kinonet_r2_best.pth"
    if len(sys.argv) > 1:
        pth = sys.argv[1]

    if not os.path.exists(pth):
        print(f"[ERROR] Model file not found: {pth}")
        print("        Run georeckon_pipeline.py first to generate the trained model.")
        sys.exit(1)

    convert_to_tflite(pth)
