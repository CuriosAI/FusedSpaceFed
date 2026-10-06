"""Only the requested component interventions; retain original Fused updates."""
import math
import statistics
import time

import torch
from fusedspacefed_core import task_loss
from research.feature_shift_digits.run_digits import DigitsClient, synchronize


class VariantClient(DigitsClient):
    def __init__(self, *args, additive=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.additive = additive

    def _joint_train(self, epochs):
        if self.additive:
            return super()._joint_train(epochs)
        self._set_trainable(encoder=True, decoder=True, classifier=True)
        self.autoencoder.train(); self.classifier.train()
        losses = []
        for _ in range(epochs):
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.ae_optimizer.zero_grad(set_to_none=True)
                self.classifier_optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = self.autoencoder(inputs)
                logits = self.classifier(reconstruction)
                loss = task_loss(logits, targets, self.task)
                loss.backward()
                self.ae_optimizer.step(); self.classifier_optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return losses

    def train_round(self, warmup_epochs, local_epochs):
        self.statistics = {}
        synchronize(self.device); started = time.perf_counter()
        self.phase = 'warmup'; warmup = self._warmup(warmup_epochs)
        synchronize(self.device); middle = time.perf_counter()
        self.phase = 'classification'; classification = self._joint_train(local_epochs)
        synchronize(self.device); ended = time.perf_counter()
        if not classification or not all(math.isfinite(v) for v in warmup + classification):
            raise FloatingPointError('Non-finite or empty classification loss')
        return {'warmup_batch_mean_loss': statistics.mean(warmup) if warmup else None,
                'classification_batch_mean_loss': statistics.mean(classification),
                'warmup_steps': len(warmup), 'classification_steps': len(classification),
                'encoder_steps': len(warmup) + len(classification),
                'decoder_steps': len(classification), 'classifier_steps': len(classification),
                'warmup_samples': len(self.loader.dataset) * warmup_epochs,
                'classification_samples': len(self.loader.dataset) * local_epochs,
                'warmup_seconds': middle - started, 'classification_seconds': ended - middle,
                'clipping': self.statistics}
