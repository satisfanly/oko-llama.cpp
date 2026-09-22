# gguf_to_safetensors.py
# python3 ./convert_gguf_to_npz.py ../Dolphin3.0-Llama3.1-8B/Dolphin-3.0-Llama-3.1-8B-F16.gguf

import sys
import torch
import numpy as np
from safetensors.torch import save_file
import gguf

# Path to your .gguf model
gguf_path = sys.argv[1]
reader = gguf.GGUFReader(gguf_path)

print(f"Found {len(reader.tensors)} tensors.")

tensor_dict = {}

for tensor in reader.tensors:
    # Convert to float32 if needed (optional; many tensors will be int4/int8)
    data = tensor.data
    name = tensor.name
    tensor_dict[name] = torch.tensor(data)
    print(f"Loaded {name}: shape={data.shape} dtype={data.dtype}")

# Save to safetensors
save_file(tensor_dict, "model_quantized.safetensors")
print("Saved to model_quantized.safetensors")
