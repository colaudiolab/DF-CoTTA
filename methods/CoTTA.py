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
    img_shape = (128, 128, 3)
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

class CoTTALearner(Learner):
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

        self.episodic = False
        self.transform = get_tta_transforms()   

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model = self.wrap_data_parallel(model)


        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.5)

        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                )
                model.train()
                for indecs, X, y,_ in tqdm(train_loader, desc="Training", leave=False, ncols=50):
                    video, audio = X
                    video = video.to(self.device, non_blocking=True)
                    audio = audio.to(self.device, non_blocking=True)
                    # X: torch.Tensor = X.to(self.device, non_blocking=True)
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    self.optimizer.zero_grad(set_to_none=True)
                    outs = model(video, audio)
                    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                    # loss1 = criterion(video_out, video_label)
                    # loss2 = criterion(audio_out, audio_label)
                    # loss3 = criterion(logits, y)
                    # loss = loss1 + loss2 + loss3
                    loss = criterion(logits, y)
                    loss.backward()
                    self.optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    model.state_dict(),
                    "model.pth"
                )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"auc: {val_meter.auc * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.auc,
                val_meter.f1_micro,
                self.optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
            if best_acc > 0.98: break
        logging_file.close()
        model.train()
        self.model = self.load_object(model, "model.pth")

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

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
        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        unlabeled_dataset = dataset.subset_at_phase(phase) 

        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=False
        )
        for indecs, X, y,_ in tqdm(unlabeled_loader, desc=desc, leave=False, ncols=50):
            video, audio = X
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            _, _, y = y
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            outputs = self.model(video, audio)['logits']
            self.model.train()
            self.model_ema.train()

            # Teacher Prediction
            anchor_prob = torch.nn.functional.softmax(self.model_anchor(video, audio)['logits'], dim=1).max(1)[0]
            standard_ema = self.model_ema(video, audio)['logits']
            # Augmentation-averaged Prediction
            N = 32
            outputs_emas = []
            to_aug = anchor_prob.mean(0)<0.92
            # to_aug = False
            if to_aug: 
                # print('augmenting...')
                for i in range(N):
                    outputs_  = self.model_ema(self.transfromVideo(video), audio)['logits'].detach()
                    outputs_emas.append(outputs_)
            # Threshold choice discussed in supplementary
            if to_aug:
                outputs_ema = torch.stack(outputs_emas).mean(0)
            else:
                outputs_ema = standard_ema
            # Augmentation-averaged Prediction
            # Student update
            loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0) 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Teacher update
            self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)
            # Stochastic restore
            if True:
                for nm, m  in self.model.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape)<0.001).float().to(self.device, non_blocking=True)
                            with torch.no_grad():
                                p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
            
            meter.record(y, outputs_ema)
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

    def transfromVideo(self, video: torch.Tensor):
        B, CT, H, W = video.shape
        video = video.view(-1, 3, H, W)
        transformed_frames = self.transform(video)
        transformed_frames = transformed_frames.view(B, CT, H, W)
        # num_frames = video.shape[0]
        # transformed_frames = []
        # for num in range(num_frames):
        #     frame = video[:,num, ...]
        #     transformed_frame = self.transform(frame).unsqueeze(0)
        #     transformed_frames.append(transformed_frame)
        # transformed_frames = torch.cat(transformed_frames, dim=0)
        
        return transformed_frames

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

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