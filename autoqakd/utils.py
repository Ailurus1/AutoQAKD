import torch
import torch.nn as nn
import time
from typing import Tuple
from torch.utils.data import DataLoader


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def estimate_memory_usage(module: nn.Module) -> float:
    total_bytes = 0

    for param in module.parameters():
        if "original_weight_tensor" in vars(param):
            element_size = vars(param)[
                "original_weight_tensor"
            ].tensor_impl.data.dtype.itemsize
        else:
            element_size = param.element_size()

        total_bytes += param.numel() * element_size

    for buffer in module.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (2**20)  # Convert to MB


def fake_quantize_embeds(
    embeds: torch.Tensor, num_bits: int = 8, signed: bool = True
) -> torch.Tensor:
    embeds_view = embeds.view(embeds.size(0), -1)

    if signed:
        q_min = -(2 ** (num_bits - 1))
        q_max = (2 ** (num_bits - 1)) - 1
    else:
        q_min = 0
        q_max = (2**num_bits) - 1

    min_vals = embeds_view.min(dim=1, keepdim=True)[0]
    max_vals = embeds_view.max(dim=1, keepdim=True)[0]

    scale = (max_vals - min_vals) / (q_max - q_min)
    scale = torch.clamp(scale, min=1e-8)

    zero_point = q_min - (min_vals / scale)

    quantized = torch.round(embeds_view / scale + zero_point)
    quantized = torch.clamp(quantized, q_min, q_max)

    return quantized


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    num_runs: int = 1,
    use_torch_compile: bool = False,
) -> Tuple[float, float]:
    if use_torch_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    model.eval()
    correct = 0
    total = 0
    total_time = 0.0

    # Warm-up run for compilation
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs, _ = model(input_ids, attention_mask)
            else:
                inputs, _ = batch
                inputs = inputs.to(device)
                if inputs.dim() > 2 and inputs.size(-1) == 784:  # MNIST-like
                    inputs = inputs.view(inputs.size(0), -1)
                outputs, _ = model(inputs)
            break

    with torch.no_grad():
        for _ in range(num_runs):
            for batch in test_loader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    start_time = time.time()
                    outputs, _ = model(input_ids, attention_mask)
                    end_time = time.time()
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)

                    if inputs.dim() > 2 and inputs.size(-1) == 784:  # MNIST-like
                        inputs = inputs.view(inputs.size(0), -1)

                    start_time = time.time()
                    outputs, _ = model(inputs)
                    end_time = time.time()

                total_time += end_time - start_time
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_time = total_time / len(test_loader.dataset) * 1000  # Convert to ms per sample

    return accuracy, avg_time


def create_data_loaders(
    train_dataset,
    test_dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    collate_fn=None,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


def get_model_stats(model: nn.Module) -> dict:
    return {
        "num_parameters": count_parameters(model),
        "memory_usage_mb": estimate_memory_usage(model),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters())
        / (2**20),
    }


def compare_models(teacher: nn.Module, student: nn.Module) -> dict:
    teacher_stats = get_model_stats(teacher)
    student_stats = get_model_stats(student)

    return {
        "teacher": teacher_stats,
        "student": student_stats,
        "compression_ratio": teacher_stats["memory_usage_mb"]
        / student_stats["memory_usage_mb"],
        "parameter_reduction": (
            teacher_stats["num_parameters"] - student_stats["num_parameters"]
        )
        / teacher_stats["num_parameters"]
        * 100,
    }
