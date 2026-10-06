import torch
from fusedspacefed_core import UNetSmallAE
from research.pathmnist_recovery.fusion_probe import scaled_decoder


def test_decoder_rescaling_is_exact_positive_additive_gain_without_encoder_change():
    torch.set_num_threads(1);torch.manual_seed(4)
    model=UNetSmallAE(3,16).eval();x=torch.rand(2,3,32,32)
    encoder=model.encoder_state();original=model.decoder_state()
    with torch.no_grad():expected,_=model(x)
    model.load_decoder_state(scaled_decoder(original,.25))
    with torch.no_grad():actual,_=model(x)
    assert torch.allclose(actual,.25*expected,atol=1e-7)
    assert all(torch.equal(v,model.encoder_state()[k]) for k,v in encoder.items())
    assert all(torch.equal(original[k],model.decoder_state()[k]) for k in original if k not in ('final.weight','final.bias'))
