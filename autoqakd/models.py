import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .core import TeacherModel, StudentModel, ModelType


class DenseTeacherModel(TeacherModel):
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list = [512, 256],
        num_classes: int = 10,
    ):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        embedding = self.features(x)
        logits = self.classifier(embedding)
        return logits, embedding


class DenseStudentModel(StudentModel):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10
    ):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.projection = nn.Linear(hidden_size, 256)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        embedding = self.features(x)
        projected_emb = self.projection(embedding)
        logits = self.classifier(embedding)
        return logits, projected_emb


class CNNTeacherModel(TeacherModel):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from torchvision.models import mobilenet_v3_large

        self.model = mobilenet_v3_large()

        self.model.features[0][0] = nn.Conv2d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False
        )

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features, num_classes
        )

    def forward(self, x):
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        embedding = torch.flatten(pooled, 1)
        logits = self.model.classifier(embedding)
        return logits, embedding


class CNNStudentModel(StudentModel):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        self.model = mobilenet_v3_small()

        self.model.features[0][0] = nn.Conv2d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False
        )

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features, num_classes
        )

        self.projection = nn.Linear(576, 960)  # 576 is MobileNetV3-small embedding size

    def forward(self, x):
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        embedding = torch.flatten(pooled, 1)
        projected_emb = self.projection(embedding)
        logits = self.model.classifier(embedding)
        return logits, projected_emb


class BertTeacherModel(TeacherModel):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits, cls_embedding


class BertStudentModel(StudentModel):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        num_layers: int = 6,
    ):
        super().__init__()
        from transformers import BertModel, BertConfig

        config = BertConfig.from_pretrained(model_name)
        config.num_hidden_layers = num_layers
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_emb = self.projection(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits, projected_emb


def get_teacher_model(
    model_type: ModelType, config: Optional[Dict[str, Any]] = None
) -> TeacherModel:
    if config is None:
        config = {}

    if model_type == ModelType.DENSE:
        return DenseTeacherModel(**config)
    elif model_type == ModelType.CNN:
        return CNNTeacherModel(**config)

    return BertTeacherModel(**config)


def get_student_model(
    model_type: ModelType, config: Optional[Dict[str, Any]] = None
) -> StudentModel:
    if config is None:
        config = {}

    if model_type == ModelType.DENSE:
        return DenseStudentModel(**config)
    elif model_type == ModelType.CNN:
        return CNNStudentModel(**config)
    elif model_type == ModelType.BERT:
        return BertStudentModel(**config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
