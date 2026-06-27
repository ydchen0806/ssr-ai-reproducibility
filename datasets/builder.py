"""Factory for building CL benchmarks."""

from .split_cifar import SplitCIFAR100, SplitCIFAR10
from .tiny_imagenet import SplitTinyImageNet
from .five_datasets import FiveDatasets
from .knowledge_editing import KnowledgeEditingBenchmark


BENCHMARK_REGISTRY = {
    "split_cifar100": SplitCIFAR100,
    "split_cifar10": SplitCIFAR10,
    "split_tiny_imagenet": SplitTinyImageNet,
    "tiny_imagenet": SplitTinyImageNet,
    "five_datasets": FiveDatasets,
    "5datasets": FiveDatasets,
    "knowledge_editing": KnowledgeEditingBenchmark,
}


def build_benchmark(config: dict):
    name = config["name"].lower()
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    cls = BENCHMARK_REGISTRY[name]
    return cls(**{k: v for k, v in config.items() if k != "name"})
