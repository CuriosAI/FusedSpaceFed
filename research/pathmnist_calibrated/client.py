"""Original two losses and private/shared split, with declared optimizations."""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from fusedspacefed_core import FusedSpaceFedClient, task_loss


class CalibratedClient(FusedSpaceFedClient):
    def __init__(self, client_id, loader, settings, device):
        super().__init__(client_id, loader, 9, 3, 16, 'multiclass', device,
                         classifier_lr=settings['classifier_lr'], autoencoder_lr=settings['autoencoder_lr'],
                         use_amp=settings['precision'] == 'fp16')
        self.settings = settings
        self.clip = settings['gradient_clip_norm']
        self.statistics = {}

    def context(self):
        precision = self.settings['precision']
        if self.device.type == 'cuda' and precision != 'fp32':
            return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)
        return nullcontext()

    def update(self, loss, joint):
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        self.ae_optimizer.zero_grad(set_to_none=True)
        self.classifier_optimizer.zero_grad(set_to_none=True)
        optimizers = [self.ae_optimizer] + ([self.classifier_optimizer] if joint else [])
        if self.scaler:
            self.scaler.scale(loss).backward()
            if self.clip is not None:
                for optimizer in optimizers:
                    self.scaler.unscale_(optimizer)
        else:
            loss.backward()
        if self.clip is not None:
            norm_ae = torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.clip, error_if_nonfinite=True)
            self.statistics.setdefault('autoencoder_unclipped_norms', []).append(float(norm_ae))
            if joint:
                norm_c = torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.clip, error_if_nonfinite=True)
                self.statistics.setdefault('classifier_unclipped_norms', []).append(float(norm_c))
        if self.scaler:
            for optimizer in optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            for optimizer in optimizers:
                optimizer.step()

    def _warmup(self, epochs):
        # Exact original implementation for the unmodified reference.
        if self.settings['precision'] == 'fp16' and self.clip is None:
            return super()._warmup(epochs)
        self.autoencoder.train(); self.classifier.eval()
        self._set_trainable(encoder=True, decoder=False, classifier=False)
        losses = []
        for _ in range(epochs):
            for x, _ in self.loader:
                x = x.to(self.device)
                with self.context():
                    d, _ = self.autoencoder(x)
                    loss = F.mse_loss(d, x)
                self.update(loss, False); losses.append(float(loss.detach()))
        return losses

    def _joint_train(self, epochs):
        if self.settings['precision'] == 'fp16' and self.clip is None:
            return super()._joint_train(epochs)
        self._set_trainable(encoder=True, decoder=True, classifier=True)
        self.autoencoder.train(); self.classifier.train()
        losses = []
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                with self.context():
                    d, _ = self.autoencoder(x)
                    loss = task_loss(self.classifier(x + d), y, self.task)
                self.update(loss, True); losses.append(float(loss.detach()))
        return losses

    def learning_rates(self, round_index):
        settings = self.settings
        factor = 1.0
        if settings['schedule'] == 'cosine':
            factor = 0.1 + 0.9 * (1 + math.cos(math.pi * (round_index - 1) / settings['schedule_horizon'])) / 2
        for group in self.classifier_optimizer.param_groups:
            group['lr'] = settings['classifier_lr'] * factor
        for group in self.ae_optimizer.param_groups:
            group['lr'] = settings['warmup_lr'] * factor
        return factor

    def train_round_at(self, round_index):
        factor = self.learning_rates(round_index)
        self.statistics = {}
        warm = self._warmup(self.settings['warmup_epochs'])
        for group in self.ae_optimizer.param_groups:
            group['lr'] = self.settings['autoencoder_lr'] * factor
        ce = self._joint_train(self.settings['local_epochs'])
        if not all(math.isfinite(v) for v in warm + ce):
            raise FloatingPointError('Non-finite phase loss')
        stats = {}
        for name, values in self.statistics.items():
            stats[name] = {'mean': sum(values) / len(values), 'max': max(values),
                           'clipped_batches': sum(v > self.clip for v in values), 'batches': len(values)}
        steps = math.ceil(self.num_samples / self.settings['batch_size'])
        return {'warmup_reconstruction_loss': sum(warm) / len(warm),
                'classification_loss': sum(ce) / len(ce),
                'warmup_steps': len(warm), 'classification_steps': len(ce),
                'learning_rate_factor': factor, 'gradient_statistics': stats,
                'expected_steps': steps * (self.settings['warmup_epochs'] + self.settings['local_epochs'])}
