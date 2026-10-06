"""Declared FusedSpaceFed variants: shared GN classifier and training-only augmentation."""
import torch
from torch import nn
from torch.nn import functional as F
from fusedspacefed_core import ResNet20V2, FusedSpaceFedClient, task_loss
from research.pathmnist_calibrated.client import CalibratedClient


def build_classifier(settings):
    model=ResNet20V2(9,3)
    normalization=settings.get('classifier_normalization','batchnorm')
    if normalization=='batchnorm':return model
    if normalization!='groupnorm8':raise ValueError('Undeclared classifier normalization')
    def replace(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,nn.BatchNorm2d):
                norm=nn.GroupNorm(8,child.num_features,eps=child.eps,affine=True)
                with torch.no_grad():norm.weight.copy_(child.weight);norm.bias.copy_(child.bias)
                setattr(parent,name,norm)
            else:replace(child)
    replace(model)
    return model


def augment_images(images,mode):
    if mode=='none':return images
    if mode!='flip-rot90':raise ValueError('Undeclared augmentation')
    if images.shape[-1]!=images.shape[-2]:raise ValueError('Rotation needs square inputs')
    result=images.clone();n=len(result)
    horizontal=torch.rand(n,device=images.device)<.5
    vertical=torch.rand(n,device=images.device)<.5
    result[horizontal]=result[horizontal].flip(-1)
    result[vertical]=result[vertical].flip(-2)
    rotations=torch.randint(0,4,(n,),device=images.device)
    for k in (1,2,3):
        selected=rotations==k
        result[selected]=torch.rot90(result[selected],k,(-2,-1))
    return result


class RecoveryClient(CalibratedClient):
    def __init__(self,client_id,loader,settings,device):
        FusedSpaceFedClient.__init__(self,client_id,loader,9,3,16,'multiclass',device,
            classifier_lr=settings['classifier_lr'],autoencoder_lr=settings['autoencoder_lr'],
            classifier=build_classifier(settings),use_amp=settings['precision']=='fp16')
        self.settings=settings;self.clip=settings['gradient_clip_norm'];self.statistics={}

    def _warmup(self,epochs):
        mode=self.settings.get('augmentation','none')
        if mode=='none':return super()._warmup(epochs)
        self.autoencoder.train();self.classifier.eval()
        self._set_trainable(encoder=True,decoder=False,classifier=False)
        losses=[]
        for _ in range(epochs):
            for x,_ in self.loader:
                x=augment_images(x.to(self.device),mode)
                with self.context():
                    d,_=self.autoencoder(x);loss=F.mse_loss(d,x)
                self.update(loss,False);losses.append(float(loss.detach()))
        return losses

    def _joint_train(self,epochs):
        mode=self.settings.get('augmentation','none')
        if mode=='none':return super()._joint_train(epochs)
        self._set_trainable(encoder=True,decoder=True,classifier=True)
        self.autoencoder.train();self.classifier.train();losses=[]
        for _ in range(epochs):
            for x,y in self.loader:
                x=augment_images(x.to(self.device),mode);y=y.to(self.device)
                with self.context():
                    d,_=self.autoencoder(x);loss=task_loss(self.classifier(x+d),y,self.task)
                self.update(loss,True);losses.append(float(loss.detach()))
        return losses
