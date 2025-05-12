import torch
import time

print("=== GPU Diagnostic ===")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test GPU speed
    print("\nRunning GPU benchmark...")
    device = torch.device('cuda')

    # Test matrix multiplication
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm up
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(50):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Matrix multiplication (4096x4096): {(end - start) / 50 * 1000:.2f} ms per operation")
    print(f"FLOPS: {2 * size**3 * 50 / (end - start) / 1e12:.2f} TFLOPs")

    # Test tensor operations
    x = torch.randn(1000, 1000, device=device)
    start = time.time()
    for _ in range(1000):
        y = torch.relu(x + 1)
    torch.cuda.synchronize()
    end = time.time()
    print(f"ReLU operations: {(end - start) / 1000 * 1000:.2f} μs per operation")

else:
    print("CUDA is not available!")

print("\n=== System Info ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CPU cores: {torch.multiprocessing.cpu_count()}")