import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationInt8WeightConfig,
)
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    FromIntXQuantizationAwareTrainingConfig,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized_intx_static
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .core import (
    QAKDConfig,
    TeacherModel,
    StudentModel,
    SimilarityLoss,
    QuantizationType,
)
from .utils import fake_quantize_embeds


class ObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: torch.nn.Module,
        weight_obs: torch.nn.Module,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs
        self.weight_obs = weight_obs

    def forward(self, input: torch.Tensor):
        observed_input = self.act_obs(input)
        observed_weight = self.weight_obs(self.weight)
        return F.linear(observed_input, observed_weight, self.bias)

    @classmethod
    def from_float(cls, float_linear, act_obs, weight_obs):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            weight_obs,
            False,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear


def insert_observers_(model, act_obs, weight_obs):
    def _is_linear(m, fqn):
        return isinstance(m, torch.nn.Linear)

    def replacement_fn(m):
        copied_act_obs = copy.deepcopy(act_obs)
        copied_weight_obs = copy.deepcopy(weight_obs)
        return ObservedLinear.from_float(m, copied_act_obs, copied_weight_obs)

    _replace_with_custom_fn_if_matches_filter(model, replacement_fn, _is_linear)


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        act_obs: torch.nn.Module,
        weight_obs: torch.nn.Module,
        weight: torch.Tensor,
        bias: torch.Tensor,
        target_dtype: torch.dtype,
    ):
        super().__init__()
        self.act_scale, self.act_zero_point = act_obs.calculate_qparams()
        weight_scale, weight_zero_point = weight_obs.calculate_qparams()
        assert weight.dim() == 2
        block_size = (1, weight.shape[1])
        self.target_dtype = target_dtype
        self.bias = bias
        self.qweight = to_affine_quantized_intx_static(
            weight, weight_scale, weight_zero_point, block_size, self.target_dtype
        )

    def forward(self, input: torch.Tensor):
        block_size = input.shape
        qinput = to_affine_quantized_intx_static(
            input,
            self.act_scale.to(input.device),
            self.act_zero_point.to(input.device),
            block_size,
            self.target_dtype,
        )
        qweight = self.qweight.to(input.device)
        bias = self.bias.to(input.device) if self.bias is not None else None
        return F.linear(qinput, qweight, bias)

    @classmethod
    def from_observed(cls, observed_linear, target_dtype):
        quantized_linear = cls(
            observed_linear.in_features,
            observed_linear.out_features,
            observed_linear.act_obs,
            observed_linear.weight_obs,
            observed_linear.weight,
            observed_linear.bias,
            target_dtype,
        )
        return quantized_linear


@dataclass
class StaticQuantConfig(AOBaseConfig):
    target_dtype: torch.dtype


@register_quantize_module_handler(StaticQuantConfig)
def _apply_static_quant(
    module: torch.nn.Module,
    config: StaticQuantConfig,
):
    return QuantizedLinear.from_observed(module, config.target_dtype)


def is_observed_linear(m, fqn):
    return isinstance(m, ObservedLinear)


def train_teacher_model(
    teacher: TeacherModel, train_loader: DataLoader, config: QAKDConfig, **kwargs
) -> TeacherModel:
    teacher = teacher.to(config.device)
    optimizer = optim.Adam(teacher.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        teacher.train()
        total_loss = 0

        for batch in tqdm(
            train_loader, desc=f"Training teacher epoch {epoch+1}/{config.epochs}"
        ):
            if config.model_type.value == "bert":
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                targets = batch["labels"].to(config.device)
                outputs, _ = teacher(input_ids, attention_mask)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(config.device), targets.to(config.device)

                if config.model_type.value == "dense" and inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)

                outputs, _ = teacher(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return teacher


def train_student_qakd(
    teacher: TeacherModel,
    student: StudentModel,
    train_loader: DataLoader,
    config: QAKDConfig,
    **kwargs,
) -> StudentModel:
    if config.do_prequant:
        assert config.qat

    student = student.to(config.device)
    teacher = teacher.to(config.device)

    activation_config = FakeQuantizeConfig(
        torch.int8,
        "per_channel" if config.model_type.value != "bert" else "per_token",
        is_symmetric=False,
    )
    weight_config = FakeQuantizeConfig(
        torch.int8, group_size=config.quantization.group_size, is_symmetric=False
    )

    if config.qat:
        quantize_(
            student,
            IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
        )

    optimizer = optim.Adam(student.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        student.train()

        for batch in tqdm(
            train_loader, desc=f"Training student epoch {epoch+1}/{config.epochs}"
        ):
            if config.model_type.value == "bert":
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                targets = batch["labels"].to(config.device)

                with torch.no_grad():
                    t_logits, t_emb = teacher(input_ids, attention_mask)
                s_logits, s_emb = student(input_ids, attention_mask)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(config.device), targets.to(config.device)

                if config.model_type.value == "dense" and inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)

                with torch.no_grad():
                    t_logits, t_emb = teacher(inputs)
                s_logits, s_emb = student(inputs)

            optimizer.zero_grad()

            task_loss = F.cross_entropy(s_logits, targets)

            distill_loss = F.kl_div(
                F.log_softmax(s_logits / config.softmax_temperature, dim=1),
                F.softmax(t_logits, dim=1),
                reduction="batchmean",
            )

            total_loss = task_loss + distill_loss

            if config.do_prequant:
                t_emb = fake_quantize_embeds(t_emb)

            if config.similarity_loss is not None:
                if config.similarity_loss == SimilarityLoss.COSINE:
                    emb_loss = nn.CosineEmbeddingLoss()(
                        s_emb,
                        t_emb,
                        target=torch.ones(s_emb.shape[0]).to(config.device),
                    )
                elif config.similarity_loss == SimilarityLoss.EUCLIDEAN:
                    emb_loss = nn.MSELoss()(s_emb, t_emb)
                elif config.similarity_loss == SimilarityLoss.MANHATTAN:
                    emb_loss = nn.L1Loss()(s_emb, t_emb)
                elif "huber" in config.similarity_loss.value:
                    delta = float(config.similarity_loss.value.split("-")[0])
                    emb_loss = nn.HuberLoss(delta=delta)(s_emb, t_emb)
                else:
                    emb_loss = 0

                total_loss += emb_loss

            total_loss.backward()
            optimizer.step()

    if config.qat:
        quantize_(student, FromIntXQuantizationAwareTrainingConfig())

    if config.quantization.quantization_type == QuantizationType.DYNAMIC:
        quantize_(
            student.eval(),
            Int8DynamicActivationInt8WeightConfig(
                act_mapping_type=MappingType.ASYMMETRIC
            ),
        )
    elif config.quantization.quantization_type == QuantizationType.STATIC:
        act_obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.int8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )

        weight_obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.int8,
            granularity=PerAxis(axis=0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )

        model_to_observe = student.module if hasattr(student, "module") else student
        insert_observers_(model_to_observe.eval(), act_obs, weight_obs)

        for batch_id, batch in enumerate(train_loader):
            if config.model_type.value == "bert":
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                s_logits, _ = model_to_observe(input_ids, attention_mask)
            else:
                inputs, _ = batch
                inputs = inputs.to(config.device)
                if config.model_type.value == "dense" and inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                s_logits, _ = model_to_observe(inputs)

            if batch_id == 10:
                break

        quantize_(student, StaticQuantConfig(torch.int8), is_observed_linear)

    return student
