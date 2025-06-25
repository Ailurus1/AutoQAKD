"""
Core classes and configurations for AutoQAKD.
"""

import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Dict, Any, Callable
import json

from .utils import estimate_memory_usage, count_parameters


class ModelType(Enum):
    """Supported model types."""
    DENSE = "dense"
    CNN = "cnn"
    BERT = "bert"


class DistillationLoss(Enum):
    KL = "KL"


class SimilarityLoss(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    HUBER_0_5 = "0.5-huber"
    HUBER_1_0 = "1.0-huber"
    HUBER_1_5 = "1.5-huber"


class QuantizationType(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    NONE = None


@dataclass
class QuantizationConfig:
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    group_size: int = 16
    activation_config: Optional[Dict[str, Any]] = None
    weight_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.activation_config is None:
            self.activation_config = {
                "dtype": torch.int8,
                "granularity": "per_channel",
                "is_symmetric": False
            }
        if self.weight_config is None:
            self.weight_config = {
                "dtype": torch.int8,
                "group_size": self.group_size,
                "is_symmetric": False
            }


@dataclass
class QAKDConfig:
    model_type: ModelType = ModelType.DENSE
    teacher_config: Optional[Dict[str, Any]] = None
    student_config: Optional[Dict[str, Any]] = None
    
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    device: str = "cuda"
    
    distillation_loss: DistillationLoss = DistillationLoss.KL
    softmax_temperature: float = 1.0
    similarity_loss: Optional[SimilarityLoss] = SimilarityLoss.COSINE
    
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    qat: bool = True
    do_prequant: bool = True
    
    seed: int = 42
    use_data_parallel: bool = False
    num_workers: int = 4
    
    def __post_init__(self):
        if self.teacher_config is None:
            self.teacher_config = {}
        if self.student_config is None:
            self.student_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = {
            "model_type": self.model_type.value,
            "teacher_config": self.teacher_config,
            "student_config": self.student_config,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "distillation_loss": self.distillation_loss.value,
            "softmax_temperature": self.softmax_temperature,
            "similarity_loss": self.similarity_loss.value if self.similarity_loss else None,
            "quantization": {
                "quantization_type": self.quantization.quantization_type.value,
                "group_size": self.quantization.group_size,
                "activation_config": self.quantization.activation_config,
                "weight_config": self.quantization.weight_config,
            },
            "qat": self.qat,
            "do_prequant": self.do_prequant,
            "seed": self.seed,
            "use_data_parallel": self.use_data_parallel,
            "num_workers": self.num_workers,
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QAKDConfig":
        config_dict = config_dict.copy()
        
        config_dict["model_type"] = ModelType(config_dict["model_type"])
        config_dict["distillation_loss"] = DistillationLoss(config_dict["distillation_loss"])
        if config_dict["similarity_loss"]:
            config_dict["similarity_loss"] = SimilarityLoss(config_dict["similarity_loss"])
        
        quantization_dict = config_dict.pop("quantization")
        config_dict["quantization"] = QuantizationConfig(
            quantization_type=QuantizationType(quantization_dict["quantization_type"]),
            group_size=quantization_dict["group_size"],
            activation_config=quantization_dict["activation_config"],
            weight_config=quantization_dict["weight_config"],
        )
        
        return cls(**config_dict)
    
    def save_pretrained(self, path: str):
        with open(f"{path}/qakd_config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "QAKDConfig":
        with open(f"{path}/qakd_config.json", "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class TeacherModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class StudentModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class QAKDModel:
    def __init__(self, config: QAKDConfig):
        self.config = config
        self.teacher = None
        self.student = None
        self._setup_seed()
        self._setup_device()
    
    def _setup_seed(self):
        import random
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
    
    def _setup_device(self):
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"
    
    def train_teacher(self, train_loader, **kwargs) -> TeacherModel:
        from .models import get_teacher_model
        
        self.teacher = get_teacher_model(self.config.model_type, self.config.teacher_config)
        
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            self.teacher = nn.DataParallel(self.teacher)
        
        self.teacher = self.teacher.to(self.config.device)
        
        return self.teacher
    
    def train_student(self, train_loader, **kwargs) -> StudentModel:
        from .models import get_student_model
        from .training import train_student_qakd
        
        if self.teacher is None:
            raise ValueError("Teacher model must be trained first. Call train_teacher() before train_student().")
        
        self.student = get_student_model(self.config.model_type, self.config.student_config)
        
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            self.student = nn.DataParallel(self.student)
        
        self.student = train_student_qakd(
            teacher=self.teacher,
            student=self.student,
            train_loader=train_loader,
            config=self.config,
            **kwargs
        )
        
        return self.student
    
    def evaluate(self, test_loader, **kwargs) -> Dict[str, float]:
        if self.student is None:
            raise ValueError("Student model must be trained first. Call train_student() before evaluate().")
        
        from .utils import evaluate_model
        
        accuracy, performance = evaluate_model(
            self.student, 
            test_loader, 
            device=self.config.device,
            **kwargs
        )
        
        return {
            "accuracy": accuracy,
            "performance_ms": performance,
            "memory_usage_mb": estimate_memory_usage(self.student),
            "num_parameters": count_parameters(self.student),
        }
    
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        self.config.save_pretrained(path)
        
        if self.teacher is not None:
            torch.save(self.teacher.state_dict(), f"{path}/teacher_model.pth")
        if self.student is not None:
            torch.save(self.student.state_dict(), f"{path}/student_model.pth")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "QAKDModel":
        from .models import get_teacher_model, get_student_model
        
        config = QAKDConfig.from_pretrained(path)
        model = cls(config)
        
        teacher_path = f"{path}/teacher_model.pth"
        if os.path.exists(teacher_path):
            teacher = get_teacher_model(config.model_type, config.teacher_config)
            if config.use_data_parallel and torch.cuda.device_count() > 1:
                teacher = nn.DataParallel(teacher)
            teacher.load_state_dict(torch.load(teacher_path, map_location=config.device))
            teacher = teacher.to(config.device)
            model.teacher = teacher
        
        student_path = f"{path}/student_model.pth"
        if os.path.exists(student_path):
            student = get_student_model(config.model_type, config.student_config)
            if config.use_data_parallel and torch.cuda.device_count() > 1:
                student = nn.DataParallel(student)
            student.load_state_dict(torch.load(student_path, map_location=config.device))
            student = student.to(config.device)
            model.student = student
        
        return model


class AutoQAKD:
    @classmethod
    def from_config(cls, config: QAKDConfig) -> QAKDModel:
        return QAKDModel(config)
    
    @classmethod
    def from_pretrained(cls, path: str) -> QAKDModel:
        return QAKDModel.from_pretrained(path)
    
    @classmethod
    def for_dense(cls, **kwargs) -> QAKDModel:
        config = QAKDConfig(model_type=ModelType.DENSE, **kwargs)
        return cls.from_config(config)
    
    @classmethod
    def for_cnn(cls, **kwargs) -> QAKDModel:
        config = QAKDConfig(model_type=ModelType.CNN, **kwargs)
        return cls.from_config(config)
    
    @classmethod
    def for_bert(cls, **kwargs) -> QAKDModel:
        config = QAKDConfig(model_type=ModelType.BERT, **kwargs)
        return cls.from_config(config) 