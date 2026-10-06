"""Fixed-width head-only IPC; no feature/label fields on the optimizer channel."""
from collections import defaultdict
import copy
import numpy as np
import torch
from torch import nn

PARAMETERS=9*64+9
FLOAT_BYTES=4
FRAME_BYTES=4  # multiprocessing Unix Connection._send_bytes for these small messages


def vector_bytes(vector):
    if vector.dtype!=torch.float32 or vector.ndim!=1 or vector.numel()!=PARAMETERS:
        raise ValueError('Only a Float32 585-parameter head vector can be sent')
    if not bool(torch.isfinite(vector).all()):raise FloatingPointError('Nonfinite head vector')
    return vector.detach().cpu().numpy().astype('<f4',copy=False).tobytes()


def read_vector(payload):
    if len(payload)!=PARAMETERS*FLOAT_BYTES:raise ValueError('Wrong head payload size')
    result=torch.from_numpy(np.frombuffer(payload,dtype='<f4').copy())
    if not bool(torch.isfinite(result).all()):raise FloatingPointError('Nonfinite head message')
    return result


def request(vector):return b'G'+vector_bytes(vector)


def reply(loss,gradient):
    loss=np.asarray([loss],dtype='<f4')
    if not bool(np.isfinite(loss).all()):raise FloatingPointError('Nonfinite local CE')
    return b'L'+loss.tobytes()+vector_bytes(gradient)


def read_reply(payload):
    if payload[:1]==b'E':raise RuntimeError(payload[1:].decode('utf8'))
    if payload[:1]!=b'L' or len(payload)!=1+(PARAMETERS+1)*FLOAT_BYTES:
        raise ValueError('Expected exactly one loss and one head gradient')
    loss=float(np.frombuffer(payload[1:5],dtype='<f4')[0]);gradient=read_vector(payload[5:])
    if not np.isfinite(loss):raise FloatingPointError('Nonfinite local CE response')
    return loss,gradient


def flatten(head):return torch.cat([p.detach().reshape(-1) for p in head.parameters()])


def install(head,vector):
    with torch.no_grad():
        head.weight.copy_(vector[:9*64].reshape(9,64));head.bias.copy_(vector[9*64:])


def head_from(vector,device):
    head=nn.Linear(64,9).to(device);install(head,vector.to(device));return head


def mean_responses(rows,device):
    if not rows:raise ValueError('All clients must participate')
    losses=torch.tensor([loss for loss,_ in rows],device=device,dtype=torch.float32)
    gradients=torch.stack([g.to(device) for _,g in rows])
    return losses.mean(),gradients.mean(0)


class WireLedger:
    def __init__(self):self.categories=defaultdict(lambda:{'messages_down':0,'messages_up':0,'payload_down':0,'payload_up':0})
    def record(self,category,direction,payload):
        row=self.categories[category];row['messages_'+direction]+=1;row['payload_'+direction]+=len(payload)
    def summary(self):
        result={}
        for name,row in self.categories.items():
            item=dict(row);item['payload_total']=row['payload_down']+row['payload_up']
            item['pipe_framing_bytes']=FRAME_BYTES*(row['messages_down']+row['messages_up'])
            item['with_pipe_framing_bytes']=item['payload_total']+item['pipe_framing_bytes'];result[name]=item
        return {'categories':result,'payload_total_bytes':sum(v['payload_total'] for v in result.values()),
                'with_pipe_framing_total_bytes':sum(v['with_pipe_framing_bytes'] for v in result.values()),
                'frame_bytes_per_message':FRAME_BYTES,'head_bytes':PARAMETERS*FLOAT_BYTES,
                'gradient_and_loss_bytes':(PARAMETERS+1)*FLOAT_BYTES,
                'per_10_client_gradient_aggregation_payload_bytes':10*(len(request(torch.zeros(PARAMETERS)))+1+(PARAMETERS+1)*FLOAT_BYTES)}


class RemoteOracle:
    """Only Connections, not local datasets, are accessible to this server API."""
    def __init__(self,pipes,device,ledger):self.pipes=pipes;self.device=device;self.ledger=ledger;self.aggregations=defaultdict(int)
    def evaluate(self,vector,category='optimization'):
        payload=request(vector)
        for pipe in self.pipes:
            pipe.send_bytes(payload);self.ledger.record(category,'down',payload)
        rows=[]
        for pipe in self.pipes:
            if not pipe.poll(180):raise TimeoutError('Head client did not reply')
            message=pipe.recv_bytes();self.ledger.record(category,'up',message);rows.append(read_reply(message))
        self.aggregations[category]+=1
        return mean_responses(rows,self.device)


def optimize(initial,oracle,lbfgs,max_iter=100,device=torch.device('cpu'),callback=None):
    """Same torch L-BFGS; closure assigns the uniform mean of local gradients."""
    head=head_from(initial,device)
    optimizer=torch.optim.LBFGS(head.parameters(),max_iter=max_iter,**lbfgs)
    trace=[]
    def closure():
        optimizer.zero_grad(set_to_none=True);theta=flatten(head)
        loss,gradient=oracle.evaluate(theta)
        head.weight.grad=gradient[:9*64].reshape(9,64).clone();head.bias.grad=gradient[9*64:].clone()
        record={'theta':theta.cpu().clone(),'gradient':gradient.cpu().clone(),'loss':float(loss.detach())}
        trace.append(record)
        if callback:callback(len(trace),record)
        return loss
    optimizer.step(closure)
    return flatten(head).cpu(),copy.deepcopy(optimizer.state_dict()),trace


def central_gradient_anchor(final_vector,optimizer_state):
    """Central previous gradient is at the point before the last accepted step."""
    state=next(iter(optimizer_state['state'].values()))
    return final_vector-state['d'].cpu()*float(state['t']),state['prev_flat_grad'].cpu(),float(state['prev_loss'])
