import torch
import torchvision.models as models
import time

def profile_memory_compute():
    device = torch.device("cuda")
    
    # 测试不同的矩阵大小
    sizes = [
        (1, 512, 512),    # 小矩阵
        (2, 512, 512),    # 中等矩阵
        (4, 512, 512),    # 中等矩阵
        (8, 512, 512),    # 中等矩阵  
        (64, 512, 512),   # 大矩阵
    ]
    
    print("Matrix Size | FP32 (ms) | FP16 (ms) | Memory BW (GB/s)")
    print("-" * 55)
    
    for batch, m, n in sizes:
        a_fp32 = torch.randn(batch, m, n).cuda()
        b_fp32 = torch.randn(batch, n, m).cuda()
        
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        
        # FP32测试
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            c_fp32 = torch.bmm(a_fp32, b_fp32)
        torch.cuda.synchronize()
        fp32_time = (time.time() - start) / 100 * 1000
        
        # FP16测试
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            c_fp16 = torch.bmm(a_fp16, b_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / 100 * 1000
        
        # 计算内存带宽需求
        bytes_fp32 = batch * m * n * 4 * 3  # input + input + output
        bytes_fp16 = batch * m * n * 2 * 3
        bandwidth_fp16 = bytes_fp16 / (fp16_time / 1000) / 1e9
        
        print(f"{batch:2d}×{m}×{n} | {fp32_time:8.3f} | {fp16_time:8.3f} | {bandwidth_fp16:10.1f}")

def profile_tensor_core_usage():
    """测试是否真正使用了Tensor Core"""
    device = torch.device("cuda")
    
    # Tensor Core友好的尺寸（16的倍数）
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print("\n=== Tensor Core Optimization ===")
    print("Batch | Shape       | FP32 (ms) | FP16 (ms) | Speedup")
    print("-" * 55)
    
    for batch_size in batch_sizes:
        # 使用Tensor Core友好的维度
        m, n, k = 1024, 1024, 1024
        
        a_fp32 = torch.randn(batch_size, m, k).cuda()
        b_fp32 = torch.randn(batch_size, k, n).cuda()
        
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        
        # 预热
        for _ in range(5):
            _ = torch.bmm(a_fp32, b_fp32)
            _ = torch.bmm(a_fp16, b_fp16)
        
        # FP32
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch.bmm(a_fp32, b_fp32)
        torch.cuda.synchronize()
        fp32_time = (time.time() - start) / 20 * 1000
        
        # FP16
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch.bmm(a_fp16, b_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / 20 * 1000
        
        speedup = fp32_time / fp16_time
        
        print(f"{batch_size:5d} | {m}×{n}×{k} | {fp32_time:8.2f} | {fp16_time:8.2f} | {speedup:6.2f}x")

if __name__ == "__main__":
    profile_memory_compute()
    profile_tensor_core_usage()