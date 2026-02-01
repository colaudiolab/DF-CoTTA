# # -*- coding: utf-8 -*-

# import torch
# from os import path
# from tqdm import tqdm
# from config import load_args, ALL_METHODS
# from models import load_backbone
# from typing import Any, Dict, List, Tuple, Optional
# from datasets import Features, load_dataset
# from utils import set_determinism, validate
# from torch._prims_common import DeviceLikeType
# from torch.utils.data import Dataset, DataLoader

# import warnings

# warnings.filterwarnings("ignore", category=UserWarning)

# def make_dataloader(
#     dataset: Dataset,
#     shuffle: bool = False,
#     batch_size: int = 256,
#     num_workers: int = 8,
#     device: Optional[DeviceLikeType] = None,
#     persistent_workers: bool = False,
# ) -> DataLoader:
#     pin_memory = (device is not None) and (torch.device(device).type == "cuda")
#     config = {
#         "batch_size": batch_size,
#         "shuffle": shuffle,
#         "num_workers": num_workers,
#         "pin_memory": pin_memory,
#         "pin_memory_device": str(device) if pin_memory else "",
#         "persistent_workers": persistent_workers,
#         "drop_last": True,
#     }
#     try:
#         from prefetch_generator import BackgroundGenerator

#         class DataLoaderX(DataLoader):
#             def __iter__(self):
#                 return BackgroundGenerator(super().__iter__())

#         return DataLoaderX(dataset, **config)
#     except ImportError:
#         return DataLoader(dataset, **config)


# def check_cache_features(root: str) -> bool:
#     files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
#     for file in files_list:
#         if not path.isfile(path.join(root, file)):
#             return False
#     return True


# @torch.no_grad()
# def cache_features(
#     backbone: torch.nn.Module,
#     dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
#     device: Optional[DeviceLikeType] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     backbone.eval()
#     X_all: List[torch.Tensor] = []
#     y_all: List[torch.Tensor] = []
#     for X, y,_ in tqdm(dataloader, "Caching"):
#         # X: torch.Tensor = backbone(X.to(device))
#         video, audio = X
#         video = video.to(device, non_blocking=True)
#         audio = audio.to(device, non_blocking=True)
#         video_label, audio_label, y = y
#         video_out, audio_out, X = backbone(video, audio)
#         y: torch.Tensor = y.to(torch.int16, non_blocking=True)
#         X_all.append(X.cpu())
#         y_all.append(y.cpu())
#     return torch.cat(X_all), torch.cat(y_all)


# def main(args: Dict[str, Any]):
#     backbone_name = args["backbone"]

#     # Select device
#     if args["cpu_only"] or not torch.cuda.is_available():
#         main_device = torch.device("cpu")
#         all_gpus = None
#     elif args["gpus"] is not None:
#         gpus = args["gpus"]
#         main_device = torch.device(f"cuda:{gpus[0]}")
#         all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
#     else:
#         main_device = torch.device("cuda:0")
#         all_gpus = None

#     if args["seed"] is not None:
#         set_determinism(args["seed"])

#     if "backbone_path" in args:
#         assert path.isfile(
#             args["backbone_path"]
#         ), f"Backbone file \"{args['backbone_path']}\" doesn't exist."
#         preload_backbone = True
#         backbone_state_dict = torch.load(
#             args["backbone_path"], map_location=main_device, weights_only=False
#         )
#         backbone, input_img_size, feature_size = load_backbone(backbone_name)
#         backbone.load_state_dict(backbone_state_dict)
#         print('loading backbone from {}'.format(args["backbone_path"]))
#     else:
#         print('Loading backbone from scratch')
#         # Load model pre-train on ImageNet if there is no base training dataset.
#         preload_backbone = False
#         load_pretrain = args["base_ratio"] == 0 or "ImageNet" not in args["dataset"]
#         backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
#         if load_pretrain:
#             assert args["dataset"] != "ImageNet", "Data may leak!!!"
#     backbone = backbone.to(main_device, non_blocking=True)
#     # print(next(backbone.parameters()).device)

#     dataset_args = {
#         "name": args["dataset"],
#         "root": args["data_root"],
#         "base_ratio": args["base_ratio"],
#         "num_phases": args["phases"],
#         "shuffle_seed": args["dataset_seed"] if "dataset_seed" in args else None,
#         'image_size': 128,
#         # 'num_classes': 4,
#         # 'txt_dir': '/mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces/race'
#         'num_classes': 2,
#         'txt_dir': '/mnt/200ssddata2t/yejianbin/multidataset'
#         # 'txt_dir': '/mnt/200ssddata2t/yejianbin/FakeAVCeleb/Classify_Based_Forieas_Methond/Forieas_Real'
#     }
#     dataset_train = load_dataset(train=True, augment=False, **dataset_args)
#     dataset_test = load_dataset(train=False, augment=False, **dataset_args)

#     # Select algorithm
#     assert args["method"] in ALL_METHODS, f"Unknown method: {args['method']}"
#     learner = ALL_METHODS[args["method"]](
#         args, backbone, feature_size, main_device, all_devices=all_gpus
#     )

#     # Load dataset
#     if args["cache_features"]:
#         if "cache_path" not in args or args["cache_path"] is None:
#             args["cache_path"] = args["saving_root"]
#         if not check_cache_features(args["cache_path"]):
#             backbone = learner.backbone.eval()
            
#             for phase in range(0, args["phases"]):
#                 train_loader = make_dataloader(
#                     dataset_train.subset_at_phase(phase),
#                     False,
#                     args["batch_size"],
#                     args["num_workers"],
#                     device=main_device,
#                 )

#                 print(path.join(args["cache_path"], f"X_{phase}.pt"))
#                 X_test, y_test = cache_features(backbone, train_loader, device=main_device)
#                 torch.save(X_test, path.join(args["cache_path"], f"dataset_X_{phase}.pt"))
#                 torch.save(y_test, path.join(args["cache_path"], f"dataset_y_{phase}.pt"))


# if __name__ == "__main__":
#     main(load_args())
#     '''
#     python main.py ACIL --dataset FakeAVCeleb --base-ratio 0.2 --phases 3 \
#     --data-root /mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces --batch-size 64 --num-workers 16 --backbone MRDF_CE \
#     --label-smoothing 0 --base-epochs 30 --learning-rate 0.001  \
#     --buffer-size 64 --IL-batch-size 64 --gpus 1 0

#     python main.py ACIL --dataset FakeAVCeleb --base-ratio 0.2 --phases 6 \
#     --data-root /mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces --batch-size 16 --num-workers 16 --backbone GAT_video_audio \
#     --label-smoothing 0 --base-epochs 4 --learning-rate 0.001  \
#     --buffer-size 64 --IL-batch-size 16 --gpus 3 2
#     '''
# # save feature based on pre-trained model
# #--------------------------------------------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader

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

@torch.no_grad()
def getFeatures(
    backbone: torch.nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[DeviceLikeType] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    CFL = ECFLoss().to(device)
    for indecs, X, y,_ in tqdm(dataloader, "Caching"):
        video, audio = X
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        video_label, audio_label, y = y
        out = backbone(video, audio)
        video_out, audio_out, X, fmaps = out['video'], out['audio'], out['features'], out['fmaps']
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        # X = CFL.ecf(fmaps)
        X_all.append(X.cpu())
        y_all.append(y.cpu())
        if len(X_all) > 10:
            break
    return torch.cat(X_all), torch.cat(y_all)

@torch.no_grad()
def incremental_features(cache_path, backbone, dataloader, main_device, phase):
    # for phase in range(0, phase):
    print(path.join(cache_path, f"X_{phase}.pt"))
    X_test, y_test = getFeatures(backbone, dataloader, device=main_device)
    torch.save(X_test, path.join(cache_path, f"MIL_X_{phase}.pt"))
    torch.save(y_test, path.join(cache_path, f"MIL_y_{phase}.pt"))