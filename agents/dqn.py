import os, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

class MLP(nn.Module):
    def __init__(self, inp, hidden, out, dueling=True):
        super().__init__()
        h1,h2=hidden
        self.body=nn.Sequential(nn.Linear(inp,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU())
        self.dueling=dueling
        if dueling:
            # Dueling DQN v - a
            self.v=nn.Linear(h2,1); self.a=nn.Linear(h2,out)
        else:
            self.head=nn.Linear(h2,out)
    def forward(self,x):
        z=self.body(x)
        if self.dueling:
            v=self.v(z); a=self.a(z); return v + (a - a.mean(dim=1,keepdim=True))
        return self.head(z)

@dataclass
class DQNConfig:
    obs_dim:int; act_dim:int; gamma:float=0.99; lr:float=1e-3; double:bool=True; dueling:bool=True; target_tau:float=0.005

class DQNAgent:
    def __init__(self, cfg:DQNConfig, hidden=(256,256), device=None):
        self.cfg=cfg
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q=MLP(cfg.obs_dim,hidden,cfg.act_dim,dueling=cfg.dueling).to(self.device)
        self.targ=MLP(cfg.obs_dim,hidden,cfg.act_dim,dueling=cfg.dueling).to(self.device)
        self.targ.load_state_dict(self.q.state_dict())  # Soft update
        self.opt=torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
    def act_eps(self, obs, eps):
        if torch.rand(())<eps: return int(torch.randint(0,self.cfg.act_dim,(1,)))   # Exploration
        with torch.no_grad():
            o=torch.tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
            q=self.q(o); return int(torch.argmax(q,dim=1).item())   # Greedy
    def update(self, batch, replay):
        import torch
        obs,act,rew,nxt,done=batch; d=self.device
        obs=torch.tensor(obs,dtype=torch.float32,device=d)
        act=torch.tensor(act,dtype=torch.int64,device=d).unsqueeze(1)
        rew=torch.tensor(rew,dtype=torch.float32,device=d).unsqueeze(1)
        nxt=torch.tensor(nxt,dtype=torch.float32,device=d)
        done=torch.tensor(done,dtype=torch.float32,device=d).unsqueeze(1)
        q=self.q(obs).gather(1,act)
        with torch.no_grad():
            if self.cfg.double:
                na=torch.argmax(self.q(nxt),dim=1,keepdim=True)
                qn=self.targ(nxt).gather(1,na)
            else:
                qn=torch.max(self.targ(nxt),dim=1,keepdim=True).values
            tgt=rew + (1.0-done)*self.cfg.gamma*qn
        loss=F.smooth_l1_loss(q,tgt)
        self.opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(),10.0); self.opt.step()
        with torch.no_grad():
            for p,tp in zip(self.q.parameters(), self.targ.parameters()):
                tp.data.mul_(1.0-self.cfg.target_tau).add_(self.cfg.target_tau*p.data)
        return float(loss.item())
    def save(self,path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"q":self.q.state_dict(),"cfg":self.cfg.__dict__}, path)
    def load(self,path):
        ck=torch.load(path,map_location=self.device); self.q.load_state_dict(ck["q"]); self.targ.load_state_dict(self.q.state_dict())
