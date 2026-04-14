import torch
import time

# CPU
device_cpu = torch.device("cpu")
a_cpu = torch.randn(2000, 2000, device=device_cpu)
b_cpu = torch.randn(2000, 2000, device=device_cpu)

start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
print("CPU time:", time.time() - start)

# MPS
if torch.backends.mps.is_available():
    device_mps = torch.device("mps")
    a_mps = torch.randn(2000, 2000, device=device_mps)
    b_mps = torch.randn(2000, 2000, device=device_mps)

    start = time.time()
    c_mps = torch.matmul(a_mps, b_mps)
    print("MPS time:", time.time() - start)