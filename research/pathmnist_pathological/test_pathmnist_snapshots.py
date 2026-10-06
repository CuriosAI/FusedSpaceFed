"""Synthetic checks; no official test data and no parameter selection."""
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from fusedspacefed_core import ResNet20V2, UNetSmallAE, clone_state_dict, weighted_average_states
from research.pathmnist_pathological.run import (
    PUBLIC, atomic_save, client_snapshot, make_client, resident_loader, restore_client,
)


def small_setup(seed=42):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config = json.loads((PUBLIC / "config.json").read_text())
    config.update(batch_size=4, use_amp=False, local_epochs=1)
    x = torch.rand(8, 3, 32, 32)
    y = torch.tensor([0, 1] * 4)
    classifier = clone_state_dict(ResNet20V2(9, 3).state_dict())
    autoencoder = clone_state_dict(UNetSmallAE(3, 16).state_dict())
    loader = resident_loader(x, y, range(8), config, 0)
    client = make_client(0, loader, config, torch.device("cpu"), classifier, autoencoder)
    return config, x, y, client


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_resident_cache_preserves_native_preprocessing_and_sampler():
    config, _, _, _ = small_setup()
    array = np.random.default_rng(7).integers(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    native = torch.stack([transform(Image.fromarray(image)) for image in array])
    labels = torch.arange(8)
    indices = [7, 2, 0, 3, 6, 5, 1, 4]
    resident = resident_loader(native, labels, indices, config, 3)
    conventional = DataLoader(Subset(TensorDataset(native, labels), indices), batch_size=4,
                              shuffle=True, generator=torch.Generator().manual_seed(420003))
    for _ in range(3):
        # Exhaust both RandomSamplers: zip alone stops before the second
        # iterator executes its final empty-permutation draw.
        resident_batches, conventional_batches = list(resident), list(conventional)
        for a, b in zip(resident_batches, conventional_batches):
            assert_same(a, b)
        assert_same(resident.generator.get_state(), conventional.generator.get_state())


def test_warmup_snapshot_preserves_classifier_bn_and_decoder():
    _, _, _, client = small_setup()
    before = client_snapshot(client)
    client._warmup(1)
    after = client_snapshot(client)
    assert_same(before["classifier"], after["classifier"])
    assert_same(before["decoder"], after["decoder"])
    assert any(not torch.equal(before["encoder"][key], after["encoder"][key]) for key in before["encoder"])
    assert not after["classifier_optimizer"]["state"]
    assert after["ae_optimizer"]["state"]
    assert any(key.endswith("running_mean") for key in before["classifier"])


def test_full_checkpoint_restore_reproduces_next_round_exactly(tmp_path):
    config, x, y, client = small_setup()
    client.train_round(1, 1)
    saved = client_snapshot(client)
    atomic_save(tmp_path / "checkpoint.pt", saved)
    loaded = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    replay = make_client(0, resident_loader(x, y, range(8), config, 0), config,
                         torch.device("cpu"), saved["classifier"], saved["autoencoder"])
    restore_client(replay, loaded)
    assert_same(client_snapshot(replay), saved)
    first = client.train_round(1, 1)
    second = replay.train_round(1, 1)
    assert_same(first, second)
    assert_same(client_snapshot(client), client_snapshot(replay))


def test_shared_aggregation_keeps_private_encoder_and_integer_bn_rule():
    config, x, y, a = small_setup()
    b = make_client(1, resident_loader(x, y, range(8), config, 1), config,
                    torch.device("cpu"), a.classifier_state(), a.autoencoder.state_dict())
    a.train_round(1, 1)
    b.train_round(1, 1)
    private_a, private_b = a.encoder_state(), b.encoder_state()
    before_a, before_b = a.classifier_state(), b.classifier_state()
    shared_c = weighted_average_states([before_a, before_b], [1, 1])
    shared_d = weighted_average_states([a.decoder_state(), b.decoder_state()], [1, 1])
    for client in (a, b):
        client.set_classifier_state(shared_c)
        client.set_decoder_state(shared_d)
    assert_same(private_a, a.encoder_state())
    assert_same(private_b, b.encoder_state())
    for name, value in shared_c.items():
        if name.endswith("num_batches_tracked"):
            assert torch.equal(value, before_a[name])
        elif value.is_floating_point():
            assert torch.allclose(value, (before_a[name] + before_b[name]) / 2)
