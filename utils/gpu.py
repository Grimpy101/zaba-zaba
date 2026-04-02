import torch


def test_gpu() -> bool:
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False
    
    device_count = torch.cuda.device_count()
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    
    print(f"Found {device_count} CUDA devices")
    print(f"Current device: {device_index} ({device_name})")
    return True