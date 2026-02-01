import torch
import numpy as np
from sklearn import metrics
import torch.nn.functional as F

class ClassificationMeter:
    def __init__(self, num_classes: int, record_logits: bool = False) -> None:
        self.num_classes = num_classes
        self.total_loss = 0.0
        self.labels = np.zeros((0,), dtype=np.int32)
        self.prediction = np.zeros((0,), dtype=np.int32)
        self.acc5_cnt = 0
        self.record_logits = record_logits
        if self.record_logits:
            self.logits = np.ndarray((0, num_classes))

    def record(self, y_true: torch.Tensor, logits: torch.Tensor) -> None:
        self.labels = np.concatenate([self.labels, y_true.cpu().numpy()])
        # Record logits
        if self.record_logits:
            logits_softmax = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            self.logits = np.concatenate([self.logits, logits_softmax])

        # Loss
        self.total_loss += float(
            torch.nn.functional.cross_entropy(logits, y_true, reduction="sum").item()
        )
        # Top-5 accuracy
        y_pred = logits.topk(self.num_classes, largest=True).indices.to(torch.int)
        acc5_judge = (y_pred == y_true[:, None]).any(dim=-1)
        self.acc5_cnt += int(acc5_judge.sum().item())

        # Record the predictions
        self.prediction = np.concatenate([self.prediction, y_pred[:, 0].cpu().numpy()])

    @property
    def accuracy(self) -> float:
        return float(metrics.accuracy_score(self.labels, self.prediction))

    @property
    def balanced_accuracy(self) -> float:
        result = metrics.balanced_accuracy_score(
            self.labels, self.prediction, adjusted=True
        )
        return float(result)

    @property
    def f1_micro(self) -> float:
        result = metrics.f1_score(self.labels, self.prediction, average="micro")
        return float(result)

    @property
    def f1_macro(self) -> float:
        result = metrics.f1_score(self.labels, self.prediction, average="macro")
        return float(result)

    @property
    def accuracy5(self) -> float:
        return self.acc5_cnt / len(self.labels)

    @property
    def loss(self) -> float:
        return float(self.total_loss / len(self.labels))
    
    @property
    def auc(self) -> float:
        return float(metrics.roc_auc_score(self.labels, self.prediction))


class MyClassificationMeter:
    def __init__(self, num_classes: int, record_logits: bool = False) -> None:
        self.num_classes = num_classes
        self.roc_lab = []
        self.roc_pre = []
        self.test_corrects = 0.0
        self.dataset_size = 0

    def record(self, y_true: torch.Tensor, logits: torch.Tensor) -> None:
        _, preds = torch.max(logits, 1)
        self.test_corrects += torch.sum(preds == y_true).to(torch.float32)

        softmax_out = torch.softmax(logits, dim=-1)
        self.roc_pre.append(softmax_out)
        self.roc_lab.append(F.one_hot(y_true, num_classes=self.num_classes)) 

        self.dataset_size += y_true.shape[0]

    @property
    def accuracy(self) -> float:
        return float(self.test_corrects / self.dataset_size)
    
    @property
    def auc(self) -> float:
        return float(metrics.roc_auc_score(torch.cat(self.roc_lab, dim=0).cpu().data, torch.cat(self.roc_pre, dim=0).cpu().data))
    
    @property
    def f1_micro(self) -> float:
        return float(0)
    
    @property
    def loss(self) -> float:
        return float(0)