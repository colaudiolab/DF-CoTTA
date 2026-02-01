# -*- coding: utf-8 -*-
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
import copy, os, math
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
        video_out, audio_out, features, fmaps,  = out['video'], out['audio'], out['features'], out['fmaps']
        x = self.fc(features)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out, 'features':features, 'fmaps':fmaps}
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

class DFCoTTALearner(Learner):
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
        self.lamb: float = args["lambda"]

        self.transform = get_tta_transforms()   

        self.symmetric_cross_entropy = SymmetricCrossEntropy(alpha=0.5)

        # get class-wise source domain prototypes
        self.proto_dir_path = os.path.join('ckpt',"prototypes")
        self.fname_proto = f"protos_{self.order}.pth"
        self.fname_proto = os.path.join(self.proto_dir_path, self.fname_proto)
        self.prototypes_src_domain_path = self.fname_proto.replace(".pth", "_domain_TIL.pth")
        self.prototypes_src_class_path = self.fname_proto.replace(".pth", "_class_TIL.pth")
        self.prototypes_src_domain, self.prototypes_src_class = None, None
        if os.path.exists(self.fname_proto.replace(".pth", "_domain_TIL.pth")):
            self.prototypes_src_domain = torch.load(self.prototypes_src_domain_path, weights_only=True)
            self.prototypes_src_class = torch.load(self.prototypes_src_class_path, weights_only=True)

        # get relative direction of source domain categories 
        self.scav_dir_path = os.path.join('ckpt', "scavs")
        self.fname_scav = f"scav_inter_{self.order}_TIL.pth"
        self.fname_scav = os.path.join(self.scav_dir_path, self.fname_scav)
        self.scav_src = None
        if os.path.exists(self.fname_scav):
            self.scav_src = torch.load(self.fname_scav, weights_only=True)
        
        self.CFL = ECFLoss(d=64).to(device)
        self.selectionRatio: float = args["ratio"]

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
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    outs = model(video, audio)
                    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
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
        
        # Select high-confidence sample for training
        samplesNum = len(unlabeled_dataset)
        selectionNum = math.ceil(self.selectionRatio * samplesNum)
        while len(unlabeled_dataset) > 0:
            unlabeled_loader = DataLoader(
                unlabeled_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=False
            )
            # select 
            prob_list = []
            for indecs, X, y,_ in tqdm(unlabeled_loader, desc='Selecting', leave=False, ncols=50):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # Teacher Prediction
                outputs = self.model_ema(video, audio)
                logits = outputs['logits']
                prob_list.append(logits)

            prob_list = torch.vstack(prob_list)
            prob_softmax = torch.nn.functional.softmax(prob_list, dim=1).max(1)[0]
            selectionNum = min(selectionNum, prob_softmax.shape[0])
            maxN = prob_softmax.shape[0] - selectionNum + 1
            threshold, _ = torch.kthvalue(prob_softmax, maxN)
            top_mask = prob_softmax >= threshold
            top_indices = torch.where(top_mask)[0]
            
            # create a pseudo-label dataset
            pseduo_dataset = unlabeled_dataset.subset(top_indices)
            # remove samples from unlabeled dataset
            unlabeled_dataset.removeSamples(top_indices)

            self.model_ema.train()
            # train the student model
            pseduo_loader = DataLoader(
                            pseduo_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=False
                            )
            for indecs, X, y    ,_ in tqdm(pseduo_loader, desc='TTA', leave=False, ncols=50):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                _, _, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)

                # sce loss
                outputs_aug = self.model(self.transfromVideo(video), audio)['logits']
                outputs_ema = self.model_ema(video, audio)['logits']

                outputs = self.model(video, audio)
                fmaps, features_domain_grad = outputs['fmaps'], outputs['fmaps']
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
                for i, c in enumerate(classes):
                    mask_class = pseudo_label == c
                    classes_features_class[i] = features_class[mask_class].mean(0)


                # Obtain the specific category of offset direction and apply constraints.
                prototypes_src_class = self.prototypes_src_class[classes].to(self.device)
                shift_direction = classes_features_class - prototypes_src_class
                shift_direction = F.normalize(shift_direction, p=2, dim=1)
                scav =  - prototypes_src_class.unsqueeze(1) + self.prototypes_src_class.unsqueeze(0).to(self.device)
                scav = F.normalize(scav, p=2, dim=2)
                loss_shifted_direction = torch.einsum("bd,bcd->bc", shift_direction, scav).mean(0).mean(0)

                # Frequency-domain alignment
                prototypes_src_domain = self.prototypes_src_domain
                prototypes_src_domain = prototypes_src_domain.to(self.device)
                prototypes_src_domain = prototypes_src_domain.view(self.nb_classes, features_domain_grad.shape[1], features_domain_grad.shape[2])
                frequency_prototypes_src_domain = self.CFL.ecf(prototypes_src_domain).mean(0)
                frequency_features_domain_grad = self.CFL.ecf(features_domain_grad).mean(0)
                loss_domain_shift = torch.mean(torch.abs(frequency_prototypes_src_domain - frequency_features_domain_grad) ** 2)


                # Student update
                loss_sce = (0.5 * self.symmetric_cross_entropy(outputs, outputs_aug)).mean(0)
                loss = (loss_sce + loss_shifted_direction * self.cav_alpha + loss_domain_shift * self.cav_beta * 5)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Teacher update
                self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
                
                # stdudent output/ teacher output
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
    
    def load_model(self, baseset_size, state_dict, train_loader=None):
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
        
        if self.prototypes_src_domain == None and self.prototypes_src_class == None:
            os.makedirs(self.proto_dir_path, exist_ok=True)
            os.makedirs(self.scav_dir_path, exist_ok=True)
            features_src_domain = torch.tensor([])
            features_src_class = torch.tensor([])
            labels_src = torch.tensor([])
            # Extracting source prototypes
            # max_length = 100000
            max_length = 1000
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
            torch.save(self.prototypes_src_domain, self.prototypes_src_domain_path)
            torch.save(self.prototypes_src_class, self.prototypes_src_class_path)
        
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
        return transformed_frames
    
    def domainFrequencyAligment(self, featureMap, y):
        # extract the frequency of the embeddings
        frequency = self.CFL.ecf(featureMap)
        base_prototype = self.prototypes_src_domain

        labels = np.array(y.cpu())
        labels_set = np.unique(labels)
        ecf_loss = 0.0
        for label in labels_set:
            index = np.where(label == labels)[0]
            frequency = featureMap[index]
            base_domain_prototype = torch.tensor(base_prototype[label], device=self.device)
            ecf_loss += self.CFL(frequency, base_domain_prototype)
        ecf_loss = ecf_loss / len(labels_set)
        return ecf_loss

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

class ECFLoss(torch.nn.Module):
    """
    ECF-Loss: 经验特征函数损失
    输入：s, T 均为 (B, N, d) 实数向量
    omega: 随机采样的一组频率向量 (N_omega, d)
    """
    def __init__(self, N_omega=256, d=64):
        super().__init__()
        # 随机初始化频率向量 (可训练或固定)
        self.omega = torch.nn.Parameter(torch.randn(N_omega, d) * 0.1, requires_grad=False)

    def ecf(self, x):
        """
        x: (B, N, d)
        return: (B, N_omega) 复数特征函数值
        """
        # x 形状 (B, N, d), omega 形状 (N_omega, d)
        # 内积 <omega, x> → (B, N, N_omega)
        inner = torch.einsum('bnd,od->bno', x, self.omega)
        # 平均特征函数 (B, N_omega) 复数
        ecf_val = torch.mean(torch.exp(1j * inner), dim=1)
        return ecf_val

    def forward(self, s, phi_T):
        """
        s, T: (B, N, d) 实数向量
        T: 已经是频域特征
        return: scalar loss
        """
        phi_s = self.ecf(s)
        # phi_T = self.ecf(T)
        # 复数差的模方 |phi_s - phi_T|^2
        loss = torch.mean(torch.abs(phi_s - phi_T) ** 2)
        return loss