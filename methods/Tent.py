# -*- coding: utf-8 -*-
import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from torch.utils.data import DataLoader
import copy
import torchvision.transforms as transforms
from .my_transforms import GaussianNoise, Clip, ColorJitterPro

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        Clip(0.0, 1.0), 
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, gaussian_std),
        Clip(clip_min, clip_max)
    ])
    return tta_transforms

def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, nb_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.nb_classes = nb_classes
        self.fc = torch.nn.Linear(backbone_output, nb_classes)

    def forward(self, video, audio, train=True, phase=0):
        out = self.backbone(video, audio)
        video_out, audio_out, x = out['video'], out['audio'], out['features']
        x = self.fc(x)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out}
        return outputs
    
    def update_fc(self, nb_classes, device):
        fc = torch.nn.Linear(self.backbone_output, nb_classes)
        if self.fc is not None:
            nb_output = self.nb_classes
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc.to(device, non_blocking=True)
        self.nb_classes = nb_classes

class TentLearner(Learner):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        data_manager,
        CL_type: str,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, data_manager, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.base_epochs: int = args["base_epochs"]
        self.CL_type: str = CL_type
        self.nb_classes: int = 0


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def learn(
        self,
        dataset,
        phase: int,
        nb_classes: int,
        meter,
        desc: str = "Incremental Learning",
    ) -> None:
        if desc == 'Re-align': return
        self.model.train()
        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        unlabeled_dataset = dataset.subset_at_phase(phase) 

        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=False
        )
        for indecs, X, y,_ in tqdm(unlabeled_loader, desc='Selecting', leave=False, ncols=50):
            video, audio = X
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            _, _, y = y
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            outputs = self.model(video, audio)['logits']

            # Student update
            loss = softmax_entropy(outputs).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            meter.record(y, outputs)
        
        return meter

    def before_validation(self, phase: int) -> None:
        self.save_object(
            self.model.state_dict(),
            f"model_{phase}.pth"
        )

    def inference(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.model(video, audio)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model
    
    def load_model(self, baseset_size, state_dict, data_loader=None):
        print('loading pretrained model')
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model.load_state_dict(state_dict)
        model = self.wrap_data_parallel(model)
        self.model = model

        params = self.model.parameters()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def update_model(self, phase=0, nb_classes=2, device="cpu"):
        if phase > 0:
            if self.CL_type == 'CIL':
                self.nb_classes = nb_classes
                self.model.update_fc(self.nb_classes, device)
        else:
            print('update model...')
            model = FusionModel(self.backbone, self.backbone_output, nb_classes).to(self.device, non_blocking=True)
            model = self.wrap_data_parallel(model)
            self.model = model
            self.model.eval()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    model_anchor = copy.deepcopy(model)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)