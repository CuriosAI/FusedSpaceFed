import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fusedspacefed_core import (
    FusedSpaceFedClient,
    ResNet20V2,
    UNetSmallAE,
    dirichlet_partition,
    pathological_partition,
    seed_everything,
    weighted_average_states,
)


def test_dirichlet_partition_assigns_every_example_once():
    labels = np.repeat(np.arange(4), 25)
    parts = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=7)
    assigned = [index for part in parts for index in part]
    assert sorted(assigned) == list(range(len(labels)))
    assert all(parts)


def test_pathological_partition_assigns_every_example_once():
    labels = np.repeat(np.arange(4), 25)
    parts = pathological_partition(labels, num_clients=4, classes_per_client=2, seed=7)
    assigned = [index for part in parts for index in part]
    assert sorted(assigned) == list(range(len(labels)))
    assert all(len(np.unique(labels[part])) <= 2 for part in parts)


def test_weighted_average_uses_requested_weights():
    result = weighted_average_states(
        [{"x": torch.tensor([1.0])}, {"x": torch.tensor([5.0])}], [3, 1]
    )
    assert torch.allclose(result["x"], torch.tensor([2.0]))


def test_bottleneck_width_and_output_shape():
    model = UNetSmallAE(in_channels=1, dz=8, base=4)
    reconstruction, latent = model(torch.rand(2, 1, 32, 32))
    assert reconstruction.shape == (2, 1, 32, 32)
    assert latent.shape == (2, 8, 8, 8)


def test_private_encoder_is_not_part_of_decoder_state():
    model = UNetSmallAE(in_channels=1, dz=8, base=4)
    assert set(model.encoder_state()).isdisjoint(model.decoder_state())
    assert all(name.startswith(UNetSmallAE.ENCODER_PREFIXES) for name in model.encoder_state())
    assert all(name.startswith(UNetSmallAE.DECODER_PREFIXES) for name in model.decoder_state())


def test_fused_joint_phase_updates_encoder_decoder_and_classifier():
    seed_everything(3)
    images = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    client = FusedSpaceFedClient(
        client_id=0,
        loader=loader,
        num_classes=2,
        in_channels=1,
        dz=4,
        task="multiclass",
        device=torch.device("cpu"),
    )
    encoder_before = {name: value.clone() for name, value in client.encoder_state().items()}
    decoder_before = {name: value.clone() for name, value in client.decoder_state().items()}
    classifier_before = {name: value.clone() for name, value in client.classifier_state().items()}
    client.train_round(warmup_epochs=0, local_epochs=1)
    assert any(not torch.equal(value, encoder_before[name]) for name, value in client.encoder_state().items())
    assert any(not torch.equal(value, decoder_before[name]) for name, value in client.decoder_state().items())
    assert any(not torch.equal(value, classifier_before[name]) for name, value in client.classifier_state().items())


def test_resnet_accepts_grayscale_and_rgb():
    assert ResNet20V2(3, 1)(torch.rand(2, 1, 32, 32)).shape == (2, 3)
    assert ResNet20V2(3, 3)(torch.rand(2, 3, 32, 32)).shape == (2, 3)
