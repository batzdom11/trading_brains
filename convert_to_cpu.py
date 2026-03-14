import torch
from pytorch_forecasting import TemporalFusionTransformer

def convert_to_cpu_recursive(obj):
    """Recursively convert all tensors to CPU"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: convert_to_cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_cpu_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_cpu_recursive(v) for v in obj)
    else:
        return obj

print("Loading original checkpoint...")
checkpoint = torch.load('models/tft_checkpoint_latest.ckpt', map_location='cpu')

print("Converting ALL tensors to CPU recursively...")
checkpoint_cpu = convert_to_cpu_recursive(checkpoint)

print("Saving CPU checkpoint...")
torch.save(checkpoint_cpu, 'models/tft_checkpoint_cpu.ckpt')

print("CPU checkpoint saved!")