import torch
from os import path
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch._prims_common import DeviceLikeType
from typing import Union, Dict, Any, Optional, Sequence
import numpy as np
loader_t = DataLoader[Union[torch.Tensor, torch.Tensor]]


class Learner(metaclass=ABCMeta):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        data_manager = None,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.args = args
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.device = device
        self.all_devices = all_devices
        self.model: torch.nn.Module
        self._known_classes = 0
        self.nb_classes = 0
        self.memory = []
        self.data_manager = data_manager
        self.CL_type: str = ""

    # @abstractmethod
    # def base_training(
    #     self,
    #     train_loader: loader_t,
    #     val_loader: loader_t,
    #     baseset_size: int,
    # ) -> None:
    #     raise NotImplementedError()

    @abstractmethod
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        phase: int,
        desc: str = "Incremental Learning"
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def before_validation() -> None:
        raise NotImplementedError()

    @abstractmethod
    def inference(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def save_object(self, model, file_name: str) -> None:
        torch.save(model, path.join(self.args["saving_root"], file_name))

    def load_object(self, model, file_name: str, file_path: str = None) -> None:
        if file_path == None:
            file_path = path.join(self.args["saving_root"], file_name)
        state_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {file_path}")
        return model

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.inference(X)

    def before_training(self, dataset) -> None:
        pass

    def _construct_exemplar(self, model, dataloader, m):
        model.eval()
        with torch.no_grad():
            class_exemplars = [[] for _ in range(self.nb_classes)]
            features = []
            labels = []
            for i, (indeces, X, y,index) in enumerate(dataloader):
                fn_img, fn_aud, label, start = index
                exemplar = list(zip(fn_img, fn_aud, label, start))
                _, _, target = y
                for j in range(len(exemplar)):
                    fn_img, fn_aud, label, start = exemplar[j]
                    label = label.item()
                    start = start.item()
                    idx = (fn_img, fn_aud, label, start)
                    class_exemplars[target[j]].append(idx)
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                feature = model(video, audio)['logits']
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
                # Clear buffer regularly
                if i % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        class_label = []
        if isinstance(m,float):
            total_num = sum([len(num) for num in class_exemplars])
            m = int(m*total_num)
            m = min(m, 500)
        if self.CL_type == 'CIL':
            labels_set = range(self._known_classes, self.nb_classes)
        total_selected_exemplars = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            data = class_exemplars[item]
            vectors = features[index]
            class_mean = np.mean(vectors, axis=0)
            prototype.append(class_mean)


            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            length_of_vectors = vectors.shape[0]
            for k in range(1, min(m//2, length_of_vectors) + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    data[i]
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                # data = np.delete(
                #     data, i, axis=0
                # )  # Remove it to avoid duplicative selection
                del data[i]
            total_selected_exemplars.extend(selected_exemplars)
        self.memory.extend(total_selected_exemplars)
        # # 随机选择m个样本作为exemplar
        # exemplars = []
        # for i, data in enumerate(dataloader):
        #     _, data, label, index = data
        #     fn_img, fn_aud, label, start = index
        #     exemplar = list(zip(fn_img, fn_aud, label, start))
        #     for i in exemplar:
        #         fn_img, fn_aud, label, start = i
        #         label = label.item()
        #         start = start.item()
        #         idx = (fn_img, fn_aud, label, start)
        #         exemplars.append(idx)
        #         if len(exemplars) >= m:
        #             exemplars = exemplars[:m]
        #             break
        # self.memory.extend(exemplars)
        print(f'Construct exemplars: {len(self.memory)}')
    
    # @abstractmethod
    def update_model(self, phase=0, nb_classes=2, device="cpu") -> None:
        pass