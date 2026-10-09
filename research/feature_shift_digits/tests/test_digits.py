"""Synthetic CPU checks; never use real Digits test images for implementation."""
import importlib.util
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from fusedspacefed_core import UNetSmallAE, clone_state_dict
from research.feature_shift_digits import data
from research.feature_shift_digits import run_digits as runner
from research.feature_shift_digits.model import DigitCNN


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('domain,shape', [('MNIST', (28,28)), ('MNIST-M', (28,28,3)), ('SVHN', (32,32,3)), ('SynthDigits', (32,32,3)), ('USPS', (16,16))])
def test_preprocessing_matches_author_sequence(domain, shape):
    images = np.random.default_rng(11).integers(0, 256, (3, *shape), dtype=np.uint8)
    stages = []
    if domain in ('SVHN','USPS','SynthDigits'):
        stages.append(transforms.Resize([28,28]))
    if domain in ('MNIST','USPS'):
        stages.append(transforms.Grayscale(num_output_channels=3))
    stages += [transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)]
    expected = torch.stack([transforms.Compose(stages)(Image.fromarray(image)) for image in images])
    cache = data.preprocess_images(images, domain)
    observed = torch.from_numpy(cache).float().div(255).sub(0.5).div(0.5)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_classifier_and_reconstruction_shapes():
    images = torch.randn(2,3,28,28)
    reconstructed, bottleneck = UNetSmallAE(3,64)(images)
    assert reconstructed.shape == images.shape
    assert bottleneck.shape == (2,64,7,7)
    model = DigitCNN()
    assert model(images + reconstructed).shape == (2,10)
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 3
    assert sum(isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) for layer in model.modules()) == 5
    with pytest.raises(ValueError):
        model(torch.randn(2,3,32,32))


@pytest.mark.local_artifacts
def test_author_model_exact_parity_if_reference_present():
    source = Path('_local/feature_shift_digits/reference/FedBN/nets/models.py')
    if not source.exists():
        pytest.skip('Author reference copy is private and optional for portable tests')
    spec = importlib.util.spec_from_file_location('reference_digit_models', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original, ours = module.DigitModel(), DigitCNN()
    ours.load_state_dict(original.state_dict())
    original.eval(); ours.eval()
    inputs = torch.randn(3,3,28,28)
    torch.testing.assert_close(ours(inputs), original(inputs), rtol=0, atol=0)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3,10)

    def forward(self, x):
        return self.fc(x.mean(dim=(2,3)))


def settings():
    return {'dz':64, 'classifier_lr':0.01, 'classifier_momentum':0.0, 'classifier_weight_decay':0.0,
            'autoencoder_lr':0.0003, 'autoencoder_betas':[0.9,0.999], 'autoencoder_eps':1e-8,
            'autoencoder_weight_decay':0.0,
            'gradient_clip_norm':1.0, 'warmup_epochs':1, 'classification_epochs':1,
            'batch_size':2, 'torch_threads':2, 'rounds':2}


def tiny_client(monkeypatch):
    monkeypatch.setattr(runner, 'DigitCNN', TinyCNN)
    images = torch.rand(4,3,28,28)*2-1
    loader = DataLoader(TensorDataset(images, torch.arange(4)), batch_size=2,
                        generator=torch.Generator().manual_seed(42))
    return runner.DigitsClient('unit', loader, settings(), torch.device('cpu')), loader


def test_two_phase_gradient_flow_and_eval(monkeypatch):
    client, loader = tiny_client(monkeypatch)
    classifier, encoder, decoder = client.classifier_state(), client.encoder_state(), client.decoder_state()
    client.phase = 'warmup'
    client._warmup(1)
    assert all(torch.equal(v, client.classifier_state()[k]) for k,v in classifier.items())
    assert all(torch.equal(v, client.decoder_state()[k]) for k,v in decoder.items())
    assert any(not torch.equal(v, client.encoder_state()[k]) for k,v in encoder.items())
    encoder = client.encoder_state()
    client.phase = 'classification'
    client._joint_train(1)
    assert any(not torch.equal(v, client.classifier_state()[k]) for k,v in classifier.items())
    assert any(not torch.equal(v, client.decoder_state()[k]) for k,v in decoder.items())
    assert any(not torch.equal(v, client.encoder_state()[k]) for k,v in encoder.items())
    before = client.classifier_state(), clone_state_dict(client.autoencoder.state_dict())
    result = runner.evaluate(client, loader)
    assert result['total'] == 4
    assert sum(map(sum, result['confusion_matrix'])) == 4
    assert sum(result['confusion_matrix'][k][k] for k in range(10)) == result['correct']
    assert result['accuracy_percent'] == 100*result['correct']/4
    assert all(torch.equal(v, client.classifier_state()[k]) for k,v in before[0].items())
    assert all(torch.equal(v, client.autoencoder.state_dict()[k]) for k,v in before[1].items())


def test_aggregation_shared_bn_and_private_exclusion():
    classifier = {'weight':torch.tensor([2.]),'bn.running_mean':torch.tensor([4.]),'bn.num_batches_tracked':torch.tensor(24)}
    second = {'weight':torch.tensor([4.]),'bn.running_mean':torch.tensor([8.]),'bn.num_batches_tracked':torch.tensor(24)}
    c,d = runner.aggregate([classifier,second], [{'final.weight':torch.tensor([2.])},{'final.weight':torch.tensor([6.])}])
    assert c['weight'].item() == 3
    assert c['bn.running_mean'].item() == 6
    assert c['bn.num_batches_tracked'].item() == 24
    assert d['final.weight'].item() == 4
    with pytest.raises(ValueError, match='Private encoder'):
        runner.aggregate([classifier], [{'enc1.weight':torch.tensor([1.])}])


def test_clipping_large_finite_gradients_and_nonfinite(monkeypatch):
    client,_ = tiny_client(monkeypatch)
    for parameter in client.classifier.parameters():
        parameter.grad = torch.full_like(parameter, 1e30)
    client.clip_gradients(client.classifier_optimizer)
    norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten().double() for p in client.classifier.parameters()]))
    assert torch.isfinite(norm) and norm <= 1.00001
    next(client.classifier.parameters()).grad.fill_(float('inf'))
    with pytest.raises(FloatingPointError):
        client.clip_gradients(client.classifier_optimizer)


def test_checkpoint_resume_exact_encoders_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'DigitCNN', TinyCNN)
    partition = tmp_path/'synthetic'
    partition.mkdir()
    generator = np.random.default_rng(7)
    for domain in data.DOMAINS:
        for split in ('train','test'):
            prefix = data.DIRECTORIES.get(domain,domain)+'-'+split
            np.save(partition/(prefix+'-images.npy'), generator.integers(0,256,(4,3,28,28),dtype=np.uint8))
            np.save(partition/(prefix+'-labels.npy'), np.arange(4,dtype=np.int64))
    monkeypatch.setattr(runner, 'verify', lambda root:{'partition_sha256':'synthetic'})
    config = {'benchmark':'unit_test_only','partition_sha256':'synthetic','run_seeds':[42,43],
              'training':settings(),'checkpoint_every':1}
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    runner.run(config, partition, full, 42, torch.device('cpu'))
    partial = runner.run(config, partition, resumed, 42, torch.device('cpu'), stop_after_round=1)
    assert partial['evaluations'] == []
    assert partial['status'] == 'running'
    runner.run(config, partition, resumed, 42, torch.device('cpu'), resume=True)
    full_state = torch.load(full/'checkpoint.pt', weights_only=False)
    resumed_state = torch.load(resumed/'checkpoint.pt', weights_only=False)
    for component in ('classifier','decoder'):
        assert all(torch.equal(v, resumed_state[component][k]) for k,v in full_state[component].items())
    for domain in data.DOMAINS:
        assert all(torch.equal(v,resumed_state['encoders'][domain][k]) for k,v in full_state['encoders'][domain].items())
    assert torch.equal(full_state['rng']['torch'], resumed_state['rng']['torch'])
    assert full_state['results']['evaluations'][0]['domains'] == resumed_state['results']['evaluations'][0]['domains']
    assert resumed_state['results']['completed_rounds'] == 2
    assert [r['round'] for r in resumed_state['results']['history']] == [1,2]
    assert len(resumed_state['results']['sessions']) == 2
    with pytest.raises(FileExistsError):
        runner.run(config, partition, full, 42, torch.device('cpu'))


def test_data_fingerprint_is_order_independent_and_sensitive():
    assert data.canonical_hash({'x':1,'y':[2]}) == data.canonical_hash({'y':[2],'x':1})
    assert data.canonical_hash({'x':1}) != data.canonical_hash({'x':2})


def fake_archive(path, count=743):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for domain in data.DOMAINS:
            directory = data.DIRECTORIES.get(domain,domain)
            shape = (16,16) if domain=='USPS' else (28,28) if domain=='MNIST' else (32,32,3) if domain in ('SVHN','SynthDigits') else (28,28,3)
            for split,n in [('train',count),('test',10)]:
                member = f'{directory}/partitions/train_part0.pkl' if split=='train' else f'{directory}/test.pkl'
                archive.writestr(member,pickle.dumps((np.zeros((n,*shape),dtype=np.uint8),np.arange(n)%10)))


def test_preparer_reuse_fingerprint_ids_and_tamper(tmp_path,monkeypatch):
    archive = tmp_path/'synthetic.zip'
    fake_archive(archive)
    monkeypatch.setattr(data,'ARCHIVE_SHA256',data.file_hash(archive))
    output = tmp_path/'prepared'
    first = data.prepare(archive,output)
    second = data.prepare(archive,output)
    assert first==second
    assert all(first['domains'][domain]['train']['count']==743 for domain in data.DOMAINS)
    train_ids = json.loads((output/'MNIST-train-ids.json').read_text())
    test_ids = json.loads((output/'MNIST-test-ids.json').read_text())
    assert train_ids[-1].endswith(':742') and not set(train_ids)&set(test_ids)
    path = output/'MNIST-train-images.npy'
    content = bytearray(path.read_bytes()); content[-1]^=1; path.write_bytes(content)
    with pytest.raises(ValueError,match='file changed'):
        data.verify(output)


def test_preparer_refuses_wrong_archive_or_unbalanced_count(tmp_path,monkeypatch):
    archive = tmp_path/'synthetic.zip'
    fake_archive(archive,count=742)
    with pytest.raises(ValueError,match='pinned'):
        data.prepare(archive,tmp_path/'wrong')
    monkeypatch.setattr(data,'ARCHIVE_SHA256',data.file_hash(archive))
    with pytest.raises(ValueError,match='743 rows'):
        data.prepare(archive,tmp_path/'unbalanced')


def test_pair_launcher_parallel_and_exit_codes(tmp_path,monkeypatch):
    from research.feature_shift_digits import launch_pair as launcher
    cfg = tmp_path/'config.json'
    cfg.write_text(json.dumps({'partition_sha256':'unit','run_seeds':[42,43],'per_seed_device':{'42':'cuda:1','43':'cuda:0'}}))
    monkeypatch.setattr(launcher,'REPO',tmp_path)
    monkeypatch.setattr(launcher,'PRIVATE',tmp_path/'private')
    monkeypatch.setattr(launcher,'CONFIG',cfg)
    monkeypatch.setattr(launcher,'verify',lambda path:{'partition_sha256':'unit'})
    def query(command,**kwargs):
        if command[0]=='nvidia-smi':
            return '0, Synthetic, 49000, 100, 0\n1, Synthetic, 49000, 100, 0\n'
        if command[1]=='status':
            return ''
        return 'unit_commit'
    monkeypatch.setattr(launcher.subprocess,'check_output',query)
    actions=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.seed=int(command[command.index('--seed')+1]); self.pid=self.seed
            self.returncode=0 if self.seed==42 else 7
            actions.append(('start',self.seed))
        def poll(self):
            actions.append(('poll',self.seed))
            return self.returncode
    monkeypatch.setattr(launcher.subprocess,'Popen',Process)
    monkeypatch.setattr(launcher.time,'sleep',lambda duration:None)
    assert launcher.main()==1
    assert actions[:2]==[('start',42),('start',43)]
    record=json.loads((tmp_path/'private/campaign.json').read_text())
    assert [row['exit_code'] for row in record['runs']]==[0,7]
    assert record['maximum_per_gpu']==1 and record['external_actions']=='none'
    with pytest.raises(FileExistsError):
        launcher.main()
