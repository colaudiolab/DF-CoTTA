# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from typing import Tuple, Iterable, Optional, Callable
from .metrics import ClassificationMeter, MyClassificationMeter


@torch.no_grad()
def validate(
    model: Callable[[torch.Tensor], torch.Tensor],
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    phase: int = 0,
    desc: Optional[str] = None
) -> ClassificationMeter:
    if isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device
    else:
        device = model.device
    meter = ClassificationMeter(num_classes)
    # meter = MyClassificationMeter(num_classes)

    for indecs, X, y,_ in tqdm(data_loader, desc=desc, leave=False):
        video, audio = X
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        # X = X.to(device, non_blocking=True)

        video_label, audio_label, y = y
        y = y.to(device, non_blocking=True)

        # Calculate the loss
        # logits: torch.Tensor = model(video, audio, train=False, phase=phase)
        outputs = model(video, audio, train=False, phase=phase)
        logits: torch.Tensor = outputs['logits']
        if isinstance(logits, tuple):
            video_out, audio_out, logits = logits
        try:
            meter.record(y, logits)
        except Exception as err:
            print(err)
            print('logits: ', logits)
            print('y: ', y)
            print('num_classes: ', num_classes)
    return meter
