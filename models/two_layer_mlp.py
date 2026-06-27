#!/usr/bin/env python3
"""
MLP网络架构 - 支持2层和4层配置

支持架构:
1. 2层 (2layer): Input (512) -> Hidden (hidden_dim) -> Classifier (output_dim)
2. 4层 (4layer): Input (512) -> 1000 -> 512 -> 256 -> Classifier (output_dim)

支持多种配置：
1. 线性分类器 (use_cosine_classifier=False): 标准Linear层 (默认)
2. 余弦分类器 (use_cosine_classifier=True): L2归一化 + 温度缩放

注意: TIL和CIL现在使用相同的网络架构（单头分类器），
区别仅在于评估时是否使用掩码屏蔽
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):
    """
    2层MLP: input_dim -> hidden_dim -> output_dim
    
    Architecture:
    - fc1: Hidden layer (input_dim -> hidden_dim) with ReLU
    - fc2: Classifier layer (hidden_dim -> output_dim)
    
    默认配置:
    - hidden_dim = 1000
    - use_cosine_classifier = False (线性分类器)
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1000, 
                 output_dim: int = 100, tau: float = 10.0,
                 use_cosine_classifier: bool = False):
        """
        Args:
            input_dim: 输入特征维度 (512 for ResNet18 features)
            hidden_dim: 隐藏层维度 (默认1000)
            output_dim: 输出类别数 (100 for CIFAR-100)
            tau: 余弦分类器的温度系数 (仅当use_cosine_classifier=True时使用)
            use_cosine_classifier: 是否使用余弦分类器
                                   False: 线性分类器 (默认)
                                   True: 余弦分类器
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.use_cosine_classifier = use_cosine_classifier
        self.arch_type = '2layer'
        
        # Hidden layer: 始终使用bias
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Classifier layer: 单头分类器（CIL和TIL都使用相同的架构）
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            logits: (batch_size, output_dim)
        """
        # Hidden layer with ReLU
        x = self.relu(self.fc1(x))
        
        # Classifier layer
        if self.use_cosine_classifier:
            # 余弦分类器: L2归一化 + 温度缩放
            return self.tau * F.linear(F.normalize(x, dim=1), 
                                       F.normalize(self.fc2.weight, dim=1))
        else:
            # 线性分类器: 标准矩阵乘法 (bias=False)
            return F.linear(x, self.fc2.weight)
    
    def get_hidden_features(self, x):
        """获取隐藏层输出（用于特征分析）"""
        with torch.no_grad():
            return self.relu(self.fc1(x))


class FourLayerMLP(nn.Module):
    """
    4层MLP: input_dim -> 1000 -> 512 -> 256 -> output_dim
    
    Architecture:
    - fc1: 512 -> 1000 with ReLU
    - fc2: 1000 -> 512 with ReLU
    - fc3: 512 -> 256 with ReLU
    - classifier: 256 -> output_dim (bias=False)
    
    用于对比2层和4层架构的性能差异
    """
    
    def __init__(self, input_dim: int = 512, output_dim: int = 100, 
                 tau: float = 10.0, use_cosine_classifier: bool = False):
        """
        Args:
            input_dim: 输入特征维度 (512 for ResNet18 features)
            output_dim: 输出类别数 (100 for CIFAR-100)
            tau: 余弦分类器的温度系数
            use_cosine_classifier: 是否使用余弦分类器
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.use_cosine_classifier = use_cosine_classifier
        self.arch_type = '4layer'
        
        # 4层架构: 512 -> 1000 -> 512 -> 256 -> output_dim
        self.fc1 = nn.Linear(input_dim, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.classifier = nn.Linear(256, output_dim, bias=False)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            logits: (batch_size, output_dim)
        """
        # 4层前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        # Classifier layer
        if self.use_cosine_classifier:
            return self.tau * F.linear(F.normalize(x, dim=1),
                                       F.normalize(self.classifier.weight, dim=1))
        else:
            return F.linear(x, self.classifier.weight)
    
    def get_hidden_features(self, x):
        """获取最后一层隐藏层输出（用于特征分析）"""
        with torch.no_grad():
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.relu(self.fc3(x))


def create_mlp_model(arch: str = '2layer', input_dim: int = 512, 
                     hidden_dim: int = 1000, output_dim: int = 100,
                     use_cosine_classifier: bool = False):
    """
    工厂函数：根据架构类型创建对应的MLP模型
    
    Args:
        arch: 架构类型 ('2layer' 或 '4layer')
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度 (仅2层架构使用)
        output_dim: 输出类别数
        use_cosine_classifier: 是否使用余弦分类器
    
    Returns:
        model: 对应的MLP模型
    """
    if arch == '2layer':
        return TwoLayerMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_cosine_classifier=use_cosine_classifier
        )
    elif arch == '4layer':
        return FourLayerMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            use_cosine_classifier=use_cosine_classifier
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Supported: '2layer', '4layer'")


if __name__ == '__main__':
    # 测试1: 2层线性分类器 (默认配置)
    print("=" * 60)
    print("Test 1: 2-Layer Linear Classifier (Default)")
    print("=" * 60)
    model = TwoLayerMLP()
    x = torch.randn(32, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Architecture: {model.arch_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Test 1 passed!")
    
    # 测试2: 4层线性分类器
    print("\n" + "=" * 60)
    print("Test 2: 4-Layer Linear Classifier")
    print("=" * 60)
    model_4layer = FourLayerMLP()
    output_4layer = model_4layer(x)
    print(f"Output shape: {output_4layer.shape}")
    print(f"Architecture: {model_4layer.arch_type}")
    print(f"Total parameters: {sum(p.numel() for p in model_4layer.parameters()):,}")
    print("✓ Test 2 passed!")
    
    # 测试3: 使用工厂函数创建模型
    print("\n" + "=" * 60)
    print("Test 3: Factory Function")
    print("=" * 60)
    model_2l = create_mlp_model(arch='2layer', hidden_dim=1000)
    model_4l = create_mlp_model(arch='4layer')
    print(f"2-layer model: {model_2l.arch_type}, params: {sum(p.numel() for p in model_2l.parameters()):,}")
    print(f"4-layer model: {model_4l.arch_type}, params: {sum(p.numel() for p in model_4l.parameters()):,}")
    print("✓ Test 3 passed!")
    
    # 测试4: 不同的Hidden维度 (2层)
    print("\n" + "=" * 60)
    print("Test 4: Different Hidden Dimensions (2-layer)")
    print("=" * 60)
    for hidden_dim in [512, 1000, 2048]:
        model_test = TwoLayerMLP(hidden_dim=hidden_dim)
        output_test = model_test(x)
        print(f"hidden_dim={hidden_dim}: output shape={output_test.shape}, params={sum(p.numel() for p in model_test.parameters()):,}")
    print("✓ Test 4 passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
