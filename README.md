# DF-CoTTA
The official code of paper "When Fake Evolves Faster than the Detector: Continual Test-time Domain Adaptation Benchmark for Multimodal Deepfake Detection"

## Train and evalute
```python
python CTTA_main.py DFCoTTA --dataset MDCDDataset --base-ratio 0.17 --phases 6 --CL-type TIL --csv-dir <dataset path> --batch-size 16 --num-workers 8 --backbone GAT_video_audio --base-epochs 5 --learning-rate 0.001 --gpus 0
```
