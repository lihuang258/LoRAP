###convert safetensors to bin
import os
import torch
from safetensors.torch import load_model, save_model,load_file
use_lora = False
def convert_safe_to_bin(sparsity_ratio,model_size,epoch):
    LORA_path = f"model.safetensors"
    LORA_weight = load_file(LORA_path, device="cpu")
    torch.save(LORA_weight,
               f"adapter_model.bin")
    import shutil
    # 源文件路径
    source_file = f"adapter_config.json"
    # 目标文件夹路径
    destination_folder = f"adapter_config.json"
    # 使用 shutil.move 移动文件
    shutil.copy(source_file, destination_folder)
    # if os.path.exists(source_file) and not os.path.exists(destination_folder):
    #     shutil.copy(source_file, destination_folder)
    print("convert finished")
