# -*- coding: utf-8 -*-
# https://github.com/mariodoebler/test-time-adaptation
import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
import torch.nn as nn
import torch.nn.functional as F
from .Learner import Learner, loader_t
from torch.utils.data import DataLoader
import copy, os
import numpy as np
import torchvision.transforms as transforms
from .my_transforms import GaussianNoise, Clip, ColorJitterPro

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (128, 239, 3)
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
        video_out, audio_out, features, fmaps, video_fmaps, audio_fmaps = out['video'], out['audio'], out['features'], out['fmaps'], out['video_fmaps'], out['audio_fmaps']
        x = self.fc(features)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out, 'features':features, 'fmaps':fmaps, 'video_fmaps':video_fmaps, 'audio_fmaps':audio_fmaps}
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

class CCoTTALearner(Learner):
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
        self.order: str = args['order']
        self.CL_type: str = CL_type
        self.nb_classes: int = 0
        self.cav_alpha = 2.0
        self.cav_beta = 0.05

        self.transform = get_tta_transforms()   

        self.symmetric_cross_entropy = SymmetricCrossEntropy(alpha=0.5)

        # get class-wise source domain prototypes
        self.proto_dir_path = os.path.join('ckpt',"prototypes")
        self.fname_proto = f"protos_{self.order}.pth"
        self.fname_proto = os.path.join(self.proto_dir_path, self.fname_proto)
        self.prototypes_src_domain, self.prototypes_src_class = None, None
        if os.path.exists(self.fname_proto.replace(".pth", "_domain.pth")):
            self.prototypes_src_domain = torch.load(self.fname_proto.replace(".pth", "_domain.pth"), weights_only=True)
            self.prototypes_src_class = torch.load(self.fname_proto.replace(".pth", "_class.pth"), weights_only=True)

        # get relative direction of source domain categories 
        self.scav_dir_path = os.path.join('ckpt', "scavs")
        self.fname_scav = f"scav_inter_{self.order}.pth"
        self.fname_scav = os.path.join(self.scav_dir_path, self.fname_scav)
        self.scav_src = None
        if os.path.exists(self.fname_scav):
            self.scav_src = torch.load(self.fname_scav, weights_only=True)

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

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

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

                    optimizer.zero_grad(set_to_none=True)
                    outs = model(video, audio)
                    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                    # loss1 = criterion(video_out, video_label)
                    # loss2 = criterion(audio_out, audio_label)
                    # loss3 = criterion(logits, y)
                    # loss = loss1 + loss2 + loss3
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
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
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
            if best_acc > 0.98: break
        logging_file.close()

        model.train()
        self.model = self.load_object(model, "model.pth")
        params = self.model.parameters()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)

        # get class-wise source domain prototypes
        if self.prototypes_src_domain == None and self.prototypes_src_class == None:
            os.makedirs(self.proto_dir_path, exist_ok=True)
            os.makedirs(self.scav_dir_path, exist_ok=True)
            features_src_domain = torch.tensor([])
            features_src_class = torch.tensor([])
            labels_src = torch.tensor([])
            # Extracting source prototypes
            # max_length = 100000
            max_length = 10
            with torch.no_grad():
                for indecs, X, y,_ in tqdm(train_loader, desc="prototypes", leave=False, ncols=50):
                    video, audio = X
                    video = video.to(self.device, non_blocking=True)
                    audio = audio.to(self.device, non_blocking=True)
                    # X: torch.Tensor = X.to(self.device, non_blocking=True)
                    video_label, audio_label, y = y
                    assert y.max() < baseset_size

                    outs = self.model(video, audio)
                    logits, tmp_features_class, tmp_features_domain = outs['logits'], outs['features'], outs['fmaps']

                    features_src_domain = torch.cat([features_src_domain, tmp_features_domain.view(tmp_features_domain.shape[0],-1).cpu()], dim=0)
                    features_src_class = torch.cat([features_src_class, tmp_features_class.view(tmp_features_class.shape[0],-1).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src_domain) > max_length:
                        break
            # create class-wise source prototypes
            self.prototypes_src_domain = torch.tensor([])
            self.prototypes_src_class = torch.tensor([])
            for i in range(self.nb_classes):
                mask = labels_src == i
                self.prototypes_src_domain = torch.cat([self.prototypes_src_domain, features_src_domain[mask].mean(dim=0, keepdim=True)], dim=0)
                self.prototypes_src_class = torch.cat([self.prototypes_src_class, features_src_class[mask].mean(dim=0, keepdim=True)], dim=0)
            torch.save(self.prototypes_src_domain, self.fname_proto.replace(".pth", "_domain.pth"))
            torch.save(self.prototypes_src_class, self.fname_proto.replace(".pth", "_class.pth"))
        
        # TODO
        if self.scav_src == None:
            self.scav_src = torch.zeros(self.nb_classes, self.nb_classes, self.prototypes_src_class.size(1))
            for i in range(self.nb_classes):
                for j in range(self.nb_classes):
                    if i != j:
                        self.scav_src[i][j] = self.prototypes_src_class[j] - self.prototypes_src_class[i]
                    else:
                        self.scav_src[i][j] = torch.zeros(1, self.prototypes_src_class.size(1))
            torch.save(self.scav_src, self.fname_scav)

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
        self.model.train()

        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        unlabeled_dataset = dataset.subset_at_phase(phase) 
        unlabeled_loader = DataLoader(
                unlabeled_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=False
            )
        for indecs, X, y,_ in tqdm(unlabeled_loader, desc='TTA', leave=False, ncols=50):
            video, audio = X
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            _, _, y = y
            y: torch.Tensor = y.to(self.device, non_blocking=True)

            # sce loss
            outputs_aug = self.model(self.transfromVideo(video), audio)['logits']
            outputs_ema = self.model_ema(video, audio)['logits']

            outputs = self.model(video, audio)
            features_domain_grad = outputs['fmaps']
            features_domain, features_class, outputs = outputs['fmaps'], outputs['features'], outputs['logits']

            # Reliable samples selection
            entropy = (- outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
            mask_entropy = entropy < (0.4 * np.log(self.nb_classes))
            features_class = features_class[mask_entropy]
            features_domain = features_domain.view(features_domain.size(0), -1)[mask_entropy]
            outputs_ = outputs[mask_entropy]


            # get class-wise target domain prototypes
            pseudo_label = outputs_.argmax(1).cpu()
            classes = torch.unique(pseudo_label)
            classes_features_class = torch.zeros(classes.size(0), features_class.size(1)).to(self.device)
            classes_features_domain = torch.zeros(classes.size(0), features_domain.size(1)).to(self.device)
            for i, c in enumerate(classes):
                mask_class = pseudo_label == c
                classes_features_class[i] = features_class[mask_class].mean(0)
                classes_features_domain[i] = features_domain[mask_class].mean(0)


            # Obtain the specific category of offset direction and apply constraints.
            prototypes_src_class = self.prototypes_src_class[classes].to(self.device)
            shift_direction = classes_features_class - prototypes_src_class
            shift_direction = F.normalize(shift_direction, p=2, dim=1)
            scav =  - prototypes_src_class.unsqueeze(1) + self.prototypes_src_class.unsqueeze(0).to(self.device)
            scav = F.normalize(scav, p=2, dim=2)
            loss_shifted_direction = torch.einsum("bd,bcd->bc", shift_direction, scav).mean(0).mean(0)


            # Obtain the offset direction of the overall domain and apply constraints.
            grad_outputs = torch.zeros_like(outputs)
            outputs_pred = outputs.argmax(dim=1)
            grad_outputs[range(outputs.shape[0]), outputs_pred] = 1
            grads = torch.autograd.grad(outputs, features_domain_grad, grad_outputs, create_graph=True, allow_unused=True)[0]
            grads = grads.view(grads.size(0), -1)
            features_domain_grad = features_domain_grad.view(features_domain_grad.size(0), -1)
            prototypes_src_domain = self.prototypes_src_domain
            prototypes_src_domain = prototypes_src_domain.to(self.device)
            prototypes_src_domain_ = prototypes_src_domain.mean(0, keepdim=True).squeeze(0)
            features_domain_grad_ = features_domain_grad.mean(0, keepdim=True).squeeze(0)
            scav_ = (prototypes_src_domain_ - features_domain_grad_)
            grads_domain = torch.einsum('bf, f -> b', grads, scav_).abs()
            loss_domain_shift = grads_domain.mean(0)

            # Student update
            loss_sce = (0.5 * self.symmetric_cross_entropy(outputs_aug, outputs_ema)).mean(0)
            loss = (loss_sce + loss_shifted_direction * self.cav_alpha + loss_domain_shift * self.cav_beta)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Teacher update
            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
            
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

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1-self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
                  - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """ Paper: https://arxiv.org/abs/1805.07836 """
    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        probs_with_correct_idx = probs.index_select(-1, targets).diag()
        return (1.0 - probs_with_correct_idx ** self.q) / self.q

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x