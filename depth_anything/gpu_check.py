"""Verify PyTorch CUDA installation"""
import torch
import platform
import sys

def check_pytorch_cuda():
    """Run PyTorch CUDA checks"""
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version.split()[0]}")
    
    print("\n=== PyTorch Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA tensor operations
        try:
            x = torch.tensor([1.0], device='cuda')
            y = x * 2
            print("\n✓ CUDA tensor operations successful")
            print(f"Tensor device: {x.device}")
            
            # Memory info
            print("\n=== GPU Memory ===")
            print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.0f} MB")
        except Exception as e:
            print(f"\n✗ CUDA tensor operations failed: {e}")
    else:
        print("\n✗ CUDA is not available")
        print("Check that PyTorch is installed with CUDA support")

if __name__ == "__main__":
    check_pytorch_cuda()