"""Original persistent optimizers/AMP; one explicit component intervention."""
import math
import time
from contextlib import nullcontext
import torch
from fusedspacefed_core import FusedSpaceFedClient,task_loss,_autocast_context


class PathClient(FusedSpaceFedClient):
    def __init__(self,cid,loader,settings,device,variant='full'):
        super().__init__(cid,loader,9,3,16,'multiclass',device,
                         classifier_lr=settings['classifier_lr'],autoencoder_lr=settings['autoencoder_lr'],
                         use_amp=settings['use_amp'])
        self.variant=variant;self.phase='warmup';self.actual={'warmup_ae':0,'classification_ae':0,'classification_classifier':0}
        self.install_step_counters()

    def install_step_counters(self):
        for name,optimizer in (('ae',self.ae_optimizer),('classifier',self.classifier_optimizer)):
            original=optimizer.step
            def step(*args,_name=name,_original=original,**kwargs):
                self.actual[self.phase+'_'+_name]+=1
                return _original(*args,**kwargs)
            optimizer.step=step

    def _joint_train(self,epochs):
        if self.variant!='decoder-only':return super()._joint_train(epochs)
        self._set_trainable(encoder=True,decoder=True,classifier=True)
        self.autoencoder.train();self.classifier.train();losses=[]
        for _ in range(epochs):
            for x,y in self.loader:
                x=x.to(self.device,non_blocking=True);y=y.to(self.device,non_blocking=True)
                self.ae_optimizer.zero_grad(set_to_none=True);self.classifier_optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device) if self.use_amp else nullcontext():
                    d,_=self.autoencoder(x);loss=task_loss(self.classifier(d),y,self.task)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward();self.scaler.step(self.ae_optimizer)
                    self.scaler.step(self.classifier_optimizer);self.scaler.update()
                else:
                    loss.backward();self.ae_optimizer.step();self.classifier_optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return losses

    def local_round(self,warmup_epochs=1,local_epochs=3):
        before=self.actual.copy();started=time.perf_counter();self.phase='warmup'
        warm=self._warmup(0 if self.variant=='no-warmup' else warmup_epochs)
        middle=time.perf_counter();self.phase='classification';ce=self._joint_train(local_epochs)
        if not ce or not all(math.isfinite(v) for v in warm+ce):raise FloatingPointError('Non-finite phase loss')
        actual={k:self.actual[k]-before[k] for k in before}
        return {'warmup_reconstruction_loss':sum(warm)/len(warm) if warm else None,
                'classification_loss':sum(ce)/len(ce),'warmup_steps':len(warm),'classification_steps':len(ce),
                'actual_optimizer_steps':actual,'warmup_seconds':middle-started,
                'classification_seconds':time.perf_counter()-middle,'client_id':self.client_id,'samples':self.num_samples}
