# -*- coding: utf-8 -*-

import torch
from os import path
from tqdm import tqdm
from config import load_args, ALL_METHODS
from models import load_backbone
from typing import Any, Dict, List, Tuple, Optional
from datasets import Features, load_dataset, DataManager
from utils import set_determinism, validate
from torch._prims_common import DeviceLikeType
from torch.utils.data import Dataset, DataLoader

import warnings
import copy
from utils.visual import incremental_features
from utils.metrics import ClassificationMeter

warnings.filterwarnings("ignore", category=UserWarning)

def make_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    batch_size: int = 256,
    num_workers: int = 8,
    device: Optional[DeviceLikeType] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    pin_memory = (device is not None) and (torch.device(device).type == "cuda")
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "pin_memory_device": str(device) if pin_memory else "",
        "persistent_workers": persistent_workers,
        "drop_last": True,
    }
    try:
        from prefetch_generator import BackgroundGenerator

        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        return DataLoaderX(dataset, **config)
    except ImportError:
        return DataLoader(dataset, **config)


def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True


@torch.no_grad()
def cache_features(
    backbone: torch.nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[DeviceLikeType] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for X, y in tqdm(dataloader, "Caching"):
        video, audio = X
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        video_out, audio_out, X = backbone(video, audio)
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        X_all.append(X.cpu())
        y_all.append(y.cpu())
    return torch.cat(X_all), torch.cat(y_all)


def main(args: Dict[str, Any]):
    backbone_name = args["backbone"]
    cache_path = 'figures'

    # Select device
    if args["cpu_only"] or not torch.cuda.is_available():
        main_device = torch.device("cpu")
        all_gpus = None
    elif args["gpus"] is not None:
        gpus = args["gpus"]
        main_device = torch.device(f"cuda:{gpus[0]}")
        all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
    else:
        main_device = torch.device("cuda:0")
        all_gpus = None

    if args["seed"] is not None:
        set_determinism(args["seed"])

    if args["backbone_path"] is not None:
        assert path.isfile(
            args["backbone_path"]
        ), f"Backbone file \"{args['backbone_path']}\" doesn't exist."
        preload_backbone = True
        # backbone, _, feature_size = torch.load(
        #     args["backbone_path"], map_location=main_device, weights_only=False
        # )
        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=False)
        state_dict = torch.load(
            args["backbone_path"], map_location=main_device, weights_only=True
        )
        # backbone.load_state_dict(state_dict)
        # print(f'loaded from {args["backbone_path"]}')
    else:
        # Load model pre-train on ImageNet if there is no base training dataset.
        preload_backbone = False
        load_pretrain = args["base_ratio"] == 0 or "ImageNet" not in args["dataset"]
        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
        if load_pretrain:
            assert args["dataset"] != "ImageNet", "Data may leak!!!"
    backbone = backbone.to(main_device, non_blocking=True)
    CL_type = args["CL_type"]


    if args["dataset"] == "MDCDDataset":
        if CL_type == 'CIL':
            csv_dir = '/Class_incremental_FakeAVCeleb'
        elif CL_type == 'DIL':
            csv_dir = '/Sequetail_incremental_DB/Domain_incremental_DB'
        elif CL_type == 'TIL':
            csv_dir = '/Sequetail_incremental_DB/Task_incremental_DB'
        csv_dir = args['csv_dir']
    else:
        csv_dir = None
        assert args["dataset"] in ["FakeAVCeleb", "multidataset", "multidataset_imbalance_1_5"], "Unknown dataset"
    dataset_args = {
        "name": args["dataset"],
        "root": args["data_root"],
        "base_ratio": args["base_ratio"],
        "num_phases": args["phases"],
        "shuffle_seed": args["dataset_seed"] if "dataset_seed" in args else None,
        'image_size': 128,
        # 'num_classes': 4,
        'num_classes': 2,
        'csv_dir': csv_dir,
        'IL_batch_size':  args["batch_size"],
        'num_workers':  args["num_workers"],
        'order': args["order"],
        'CL_type': CL_type,
        'shot': args["shot"]
    }
    data_manager = DataManager(**dataset_args)
    # dataset_train = load_dataset(train=True, augment=False, **dataset_args)
    # dataset_test = load_dataset(train=False, augment=False, **dataset_args)
    dataset_train = data_manager.get_dataset(train=True)
    dataset_test = data_manager.get_dataset(train=False)

    # Select algorithm
    assert args["method"] in ALL_METHODS, f"Unknown method: {args['method']}"
    learner = ALL_METHODS[args["method"]](
        args=args, backbone=backbone, backbone_output=feature_size, data_manager=data_manager, CL_type=CL_type, device=main_device, all_devices=all_gpus
    )

    # Base training
    if args["base_ratio"] > 0:
        train_subset = dataset_train.subset_at_phase(0)
        test_subset = dataset_test.subset_at_phase(0)
        train_loader = make_dataloader(
            train_subset,
            True,
            args["batch_size"],
            args["num_workers"],
            device=main_device,
        )
        test_loader = make_dataloader(
            test_subset,
            False,
            args["batch_size"],
            args["num_workers"],
            device=main_device,
        )
        total_classes = dataset_train.num_classes
        if preload_backbone:
            learner.load_model(total_classes, state_dict, train_loader)
        else:
            learner.base_training(
                train_loader,
                test_loader,
                # dataset_train.base_size,
                dataset_train.num_classes,
            )

        # incremental_features(cache_path, learner.backbone, test_loader, main_device, phase=0)

    # Incremental learning
    log_file_path = path.join(args["saving_root"], "IL.csv")
    log_file = open(log_file_path, "w", buffering=1)
    print(
        "phase", "acc@avg", "af", file=log_file, sep=","
    )

    base_learner = copy.deepcopy(learner)
    num = 3
    if CL_type == 'TIL':
        all_acc = torch.zeros((num, args["phases"], args["phases"]))
        all_auc = torch.zeros((num, args["phases"], args["phases"]))
    elif CL_type == 'CIL':
        all_acc = torch.zeros((num, args["phases"]))
        all_af = torch.zeros((num, args["phases"]))
    elif CL_type == 'DIL':
        all_acc = torch.zeros((num, args["phases"]))
        all_auc = torch.zeros((num, args["phases"]))
    for seed in range(num):
        learner = copy.deepcopy(base_learner)
        set_seed(seed)
        dataset_train = data_manager.get_dataset(train=True)
        total_classes = dataset_train.num_classes
        
        print('start incremental learning')
        first_acc = []
        incremental_dataset_train = copy.deepcopy(dataset_train)
        for phase in range(0, args["phases"]):
            print(f"Phase {phase}")
            if CL_type == 'CIL':
                total_classes += 1 if phase > 0 else 0

            if CL_type == 'TIL':
                test_subsets = []
                for p in range(phase + 1):
                    test_subsets.append(dataset_test.subset_at_phase(p))
                test_loaders = []
                for p in range(phase + 1):
                    test_loader = make_dataloader(
                        test_subsets[p],
                        False,
                        args["IL_batch_size"],
                        args["num_workers"],
                        device=main_device,
                    )
                    test_loaders.append(test_loader)

            meter = None
            if phase == 0:
                learner.learn(incremental_dataset_train, dataset_train.base_size, phase, total_classes, "Re-align")
            else:
                meter = ClassificationMeter(total_classes)
                learner.learn(dataset_test, phase, total_classes, meter, "Test-time Adaptation")
                print(
                    phase,
                    f"{meter.accuracy*100:.2f}",
                    f"{meter.auc*100:.2f}",
                    file=log_file,
                    sep=",",
                )
                print(
                    f"acc@1: {meter.accuracy * 100:.2f}%",
                    f"auc: {meter.auc * 100:.2f}%",
                    sep="    ",
                )
                # incremental_features(cache_path, learner.backbone, test_loaders[phase], main_device, phase)
            learner.before_validation(phase)


            # Validation
            union_test_subset = dataset_test.subset_until_phase(phase)
            # print(f'phase: {phase}, union test subset: {len(union_test_subset)}, dataset test: {len(dataset_test)}')
            union_test_loader = make_dataloader(
                union_test_subset,
                False,
                args["IL_batch_size"],
                args["num_workers"],
                device=main_device,
            )
            AF = 0.0
            if CL_type == 'TIL':
                sum_acc = 0.0
                val_meters = []
                for p in range(phase + 1):
                    if p == phase and meter is not None:
                        val_meter = meter
                    else:
                        # performance on all previous task
                        val_meter = validate(
                            learner.model,
                            test_loaders[p],
                            total_classes,
                            p,
                            desc=f"Phase {p}",
                        )
                    val_meters.append(val_meter)
                    sum_acc += val_meter.accuracy
                    all_acc[seed, phase, p] = val_meter.accuracy
                    all_auc[seed, phase, p] = val_meter.auc
                    if phase > 0 and p < phase:
                        # average forgetting factor
                        AF += val_meter.accuracy - first_acc[p]
                    print(
                        f"acc@1: {val_meter.accuracy * 100:.2f}%",
                        f"auc: {val_meter.auc * 100:.2f}%",
                        sep="    ",
                    )
                    print(
                        phase,
                        f"{val_meter.accuracy*100:.2f}",
                        f"{val_meter.auc*100:.2f}",
                        file=log_file,
                        sep=",",
                    )

                # first acc of each task at the last test datasubset
                first_acc.append(val_meter.accuracy)
                AF = AF / phase if phase > 0 else 0.0
                print(
                    f"acc@avg: {sum_acc / (phase + 1) * 100:.2f}%",
                    f"AF: {AF * 100:.2f}%",
                    sep="    ",
                )
                print(
                    phase,
                    f"{sum_acc / (phase + 1) * 100:.2f}",
                    f"{AF * 100:.2f}",
                    "\n",
                    file=log_file,
                    sep=" ",
                )
            else:
                
                # performance on all previous task
                val_meter = validate(
                    learner.model,
                    union_test_loader,  # historical union test set
                    total_classes,
                    phase,
                    desc=f"Phase {phase}",
                )

                sum_acc = val_meter.accuracy
                
                
                if phase > 0:
                    # average forgetting factor
                    AF = sum_acc - first_acc[0]
                all_acc[seed,phase] = sum_acc
                if CL_type == 'CIL':
                    AUC = 0
                    all_af[seed,phase] = AF
                else:
                    AUC = val_meter.auc
                    all_auc[seed, phase] = AUC
                print(
                    f"acc@1: {sum_acc * 100:.2f}%",
                    f"auc: {AUC * 100:.2f}%",
                    f"AF: {AF * 100:.2f}%",
                    sep="    ",
                )
                print(
                    phase,
                    f"{sum_acc * 100:.2f}",
                    f"{AUC * 100:.2f}",
                    f"{AF * 100:.2f}",
                    "\n",
                    file=log_file,
                    sep=" ",
                )
                # first acc of each task at the last test datasubset
                first_acc.append(sum_acc)
            sum_acc = 0.0
            AF = 0.0

    
    if CL_type == 'TIL' or CL_type == 'DIL':
        acc_average = all_acc.mean(0)*100
        auc_average = all_auc.mean(0)*100
        acc_std = all_acc.std(0)*100
        auc_std = all_auc.std(0)*100
        print(f"acc_average: {acc_average}", file=log_file, sep=" ")
        print(f"auc_average: {auc_average}", file=log_file, sep=" ")
        print(f"acc_std: {acc_std}", file=log_file, sep=" ")
        print(f"auc_std: {auc_std}", file=log_file, sep=" ")
    elif CL_type == 'CIL':
        acc_average = all_acc.mean(0)*100
        af_average = all_af.mean(0)*100
        acc_std = all_acc.std(0)*100
        af_std = all_af.std(0)*100
        print(f"acc_average: {acc_average}", file=log_file, sep=" ")
        print(f"af_average: {af_average}", file=log_file, sep=" ")
        print(f"acc_std: {acc_std}", file=log_file, sep=" ")
        print(f"af_std: {af_std}", file=log_file, sep=" ")


    log_file.close()
    

import random
import numpy as np
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    import time
    start_time = time.time()
    
    set_seed(0)
    main(load_args())

    end_time = time.time()
    print("Time used:", end_time - start_time)
