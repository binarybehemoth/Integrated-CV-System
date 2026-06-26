"""Confirm PyTorch reaches the GPU. Run: python scripts/verify_gpu.py"""
import torch

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version    :", torch.version.cuda)
    print("Device name     :", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3, device="cuda")
    y = x @ x
    print("GPU matmul OK   :", y.shape, "on", y.device)
else:
    print("Running on CPU. This is fine; training will be slow.")
