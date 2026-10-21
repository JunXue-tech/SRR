"""Three teacher-to-student objectives; all teacher targets are detached."""
import torch
from torch import nn
from torch.nn import functional as F

class PredictionDistillation(nn.Module):
    def __init__(self, temperature=2.):
        super().__init__()
        if temperature <= 0: raise ValueError("Temperature must be positive")
        self.temperature=temperature
    def forward(self, r):
        t=self.temperature
        p=r['teacher_logit'].detach().float()/t
        q=r['student_logit'].float()/t
        lp=torch.stack([F.logsigmoid(-p),F.logsigmoid(p)],-1)
        lq=torch.stack([F.logsigmoid(-q),F.logsigmoid(q)],-1)
        return t*t*(lp.exp()*(lp-lq)).sum(-1)

class RoutingDistillation(nn.Module):
    """Read-distribution KL plus temporal-pooling KL, with unit weights."""
    def forward(self, r):
        valid=r['valid'].float(); count=valid.sum(1).clamp_min(1)
        p=r['teacher_reads'].detach().float().clamp_min(1e-8)
        q=r['student_reads'].float().clamp_min(1e-8)
        read=((p*(p.log()-q.log())).sum(-1)*valid[:,None]).sum((1,2))/(p.shape[1]*count)
        p=r['teacher_attention'].detach().float().clamp_min(1e-8)
        q=r['student_attention'].float().clamp_min(1e-8)
        temporal=(p*(p.log()-q.log())*valid).sum(1)
        return read+temporal

class ResponseDistillation(nn.Module):
    """Per-frame cosine relations across six features; off-diagonal MSE."""
    def forward(self, r):
        valid=r['valid'].float();count=valid.sum(1).clamp_min(1)
        t=F.normalize(r['teacher_tokens'].detach().float(),dim=-1)
        s=F.normalize(r['student_tokens'].float(),dim=-1)
        delta=(s@s.transpose(-1,-2)-t@t.transpose(-1,-2)).square()
        off=~torch.eye(t.shape[2],device=valid.device,dtype=torch.bool)
        return (delta[...,off].mean(-1)*valid).sum(1)/count

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.):
        super().__init__()
        self.prediction=PredictionDistillation(temperature)
        self.routing=RoutingDistillation()
        self.response=ResponseDistillation()
    def forward(self, result):
        p=self.prediction(result);r=self.routing(result);f=self.response(result)
        return p+r+f,dict(prediction_kd=p,routing_kd=r,response_kd=f)
