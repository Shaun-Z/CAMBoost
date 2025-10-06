import torch
import torchvision.models as models
import time

def identify_root_cause():
    """精确定位FP16性能劣化的主导因素"""
    
    device = torch.device("cuda")
    
    print("=== 逐一排查根本原因 ===")
    
    # 测试1: Tensor Core利用率问题
    print("\n1. 测试Tensor Core利用率:")
    
    # 小矩阵 (类似ResNet卷积)
    small_a = torch.randn(1, 256, 256).to(device)
    small_b = torch.randn(1, 256, 256).to(device)

    # 大矩阵 (能充分利用Tensor Core)
    large_a = torch.randn(1, 2048, 2048).to(device)
    large_b = torch.randn(1, 2048, 2048).to(device)
    
    # 小矩阵测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.bmm(small_a, small_b)
    torch.cuda.synchronize()
    small_fp32 = (time.time() - start) / 100 * 1000
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.bmm(small_a.half(), small_b.half())
    torch.cuda.synchronize()
    small_fp16 = (time.time() - start) / 100 * 1000
    
    # 大矩阵测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = torch.bmm(large_a, large_b)
    torch.cuda.synchronize()
    large_fp32 = (time.time() - start) / 10 * 1000
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = torch.bmm(large_a.half(), large_b.half())
    torch.cuda.synchronize()
    large_fp16 = (time.time() - start) / 10 * 1000
    
    print(f"   小矩阵(256x256): FP32={small_fp32:.3f}ms, FP16={small_fp16:.3f}ms, 差异={small_fp16-small_fp32:.3f}ms")
    print(f"   大矩阵(2048x2048): FP32={large_fp32:.3f}ms, FP16={large_fp16:.3f}ms, 差异={large_fp16-large_fp32:.3f}ms")
    
    if small_fp16 > small_fp32 and large_fp16 < large_fp32:
        print(f"   ✓ 确认: Tensor Core利用率是主要问题!")
        tensor_core_issue = True
    else:
        tensor_core_issue = False
    
    # 测试2: GPU调度开销
    print("\n2. 测试GPU调度开销:")
    
    # 单个大kernel vs 多个小kernel
    big_tensor = torch.randn(1, 1024, 1024).cuda()
    small_tensors = [torch.randn(1, 128, 128).cuda() for _ in range(64)]  # 64个小tensor
    
    # 大kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = torch.relu(big_tensor)
    torch.cuda.synchronize()
    big_kernel_time = (time.time() - start) / 50 * 1000
    
    # 多个小kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        for small_tensor in small_tensors:
            _ = torch.relu(small_tensor)
    torch.cuda.synchronize()
    small_kernel_time = (time.time() - start) / 50 * 1000
    
    print(f"   大kernel: {big_kernel_time:.3f}ms")
    print(f"   64个小kernel: {small_kernel_time:.3f}ms")
    print(f"   调度开销: {small_kernel_time - big_kernel_time:.3f}ms")
    
    # 测试3: 内存访问模式
    print("\n3. 测试内存访问模式:")
    
    # 连续访问
    continuous = torch.randn(1, 3, 224, 224).cuda()
    
    # 非连续访问
    non_continuous = torch.randn(1, 224, 224, 3).cuda().permute(0, 3, 1, 2)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = continuous.half()
    torch.cuda.synchronize()
    continuous_time = (time.time() - start) / 1000 * 1000
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = non_continuous.half()
    torch.cuda.synchronize()
    non_continuous_time = (time.time() - start) / 1000 * 1000
    
    print(f"   连续内存访问: {continuous_time:.4f}ms")
    print(f"   非连续内存访问: {non_continuous_time:.4f}ms")
    
    # 结论
    print(f"\n=== 结论 ===")
    if tensor_core_issue:
        print(f"主导因素: Tensor Core利用率低")
        print(f"原因: ResNet101在batch_size=1时，矩阵运算规模太小")
        print(f"解决方案: 增加batch_size或使用FP32")
    else:
        print(f"主导因素: 需要进一步分析")

if __name__ == "__main__":
    identify_root_cause()