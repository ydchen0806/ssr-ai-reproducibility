#!/usr/bin/env python3
"""
Bio-inspired Competitive Sparsity (SSR) 正则化实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_bio_cs(weights: torch.Tensor, 
                   A_exc: float = 1.0, 
                   A_inh: float = 0.8,
                   sigma_exc: float = 0.2, 
                   sigma_inh: float = 0.5) -> torch.Tensor:
    """
    计算SSR正则化项
    
    使用Mexican Hat函数促进权重向量的竞争稀疏性
    
    Args:
        weights: 权重矩阵 (output_dim, input_dim)
        A_exc: 兴奋强度
        A_inh: 抑制强度
        sigma_exc: 兴奋范围
        sigma_inh: 抑制范围
    
    Returns:
        SSR正则化损失
    """
    # 归一化权重
    w_norm = F.normalize(weights, dim=1)
    
    # 计算余弦相似度
    cos_sim = torch.clamp(torch.matmul(w_norm, w_norm.T), -1.0, 1.0)
    
    # 转换为距离
    distance = torch.sqrt(torch.clamp(1 - cos_sim, min=1e-8))
    
    # Mexican Hat函数
    inh = A_inh * torch.exp(-(distance ** 2) / (2 * sigma_inh ** 2))
    exc = A_exc * torch.exp(-(distance ** 2) / (2 * sigma_exc ** 2))
    
    # 竞争稀疏性矩阵
    P = (inh - exc) + (A_exc - A_inh)
    
    # 移除对角线（自身比较）
    P = P - torch.diag(torch.diag(P))
    
    # 计算平均竞争强度
    n = weights.size(0)
    return torch.sum(P) / (n * (n - 1) + 1e-8)


class BioCS:
    """
    SSR正则化包装器
    """
    
    def __init__(self, lambda_biocs: float = 0.001, **kwargs):
        """
        Args:
            lambda_biocs: SSR正则化强度
            **kwargs: 传递给compute_bio_cs的其他参数
        """
        self.lambda_biocs = lambda_biocs
        self.kwargs = kwargs
    
    def penalty(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算SSR惩罚项
        
        Args:
            weights: 权重矩阵
        
        Returns:
            SSR正则化损失
        """
        return self.lambda_biocs * compute_bio_cs(weights, **self.kwargs)


if __name__ == '__main__':
    # 测试SSR
    print("Testing SSR...")
    
    # 创建随机权重
    weights = torch.randn(100, 256)
    
    # 计算SSR
    loss = compute_bio_cs(weights)
    print(f"SSR loss: {loss.item():.6f}")
    
    # 测试BioCS类
    biocs = BioCS(lambda_biocs=0.001)
    penalty = biocs.penalty(weights)
    print(f"SSR penalty (with lambda): {penalty.item():.6f}")
    
    print("\n✓ SSR test passed!")
