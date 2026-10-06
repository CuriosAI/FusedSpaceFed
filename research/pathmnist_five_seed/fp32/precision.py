"""Separate precision-only campaign; historical scientific sources stay immutable."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / 'research/pathmnist_five_seed/fp32'
PRIVATE = ROOT / '_local/pathmnist_five_seed/fp32'


def force_fp32():
    # Disable reduced-mantissa CUDA arithmetic as well as AMP. No change to
    # architecture, optimizer, learning rate, losses or train/test protocol.
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def check_client_precision(client):
    if client.use_amp or client.scaler is not None:
        raise AssertionError('FP32 campaign must not create an AMP scaler')
    for model in (client.classifier, client.autoencoder):
        if any(v.is_floating_point() and v.dtype != torch.float32
               for v in model.state_dict().values()):
            raise AssertionError('Non-FP32 model state')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise AssertionError('TF32 enabled in FP32 campaign')
