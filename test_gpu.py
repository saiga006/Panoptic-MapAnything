#!/usr/bin/env python3
"""
Quick GPU test script to verify CUDA availability
"""
import torch
import sys

print("="*80)
print("GPU AVAILABILITY TEST")
print("="*80)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n✓ CUDA IS AVAILABLE!")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")
    
    # Test tensor creation on GPU
    print(f"\nTesting GPU tensor operations...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"  ✓ GPU tensor operations successful!")
        print(f"  Result shape: {z.shape}")
        print(f"  Device: {z.device}")
    except Exception as e:
        print(f"  ✗ GPU tensor operations failed: {e}")
        sys.exit(1)
    
    sys.exit(0)
else:
    print(f"\n✗ CUDA NOT AVAILABLE!")
    print(f"  This could be due to:")
    print(f"  1. PyTorch not compiled with CUDA support")
    print(f"  2. CUDA drivers not installed")
    print(f"  3. No GPU allocated by SLURM")
    print(f"  4. Wrong conda environment")
    sys.exit(1)

print("="*80)
