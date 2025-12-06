import numpy as np
from dataclasses import dataclass, field
from typing import List
from collections import OrderedDict

@dataclass
class Preset:
    promote: int
    prefetch: int
    hot_q: float
    stride: int

@dataclass
class EnvConfig:
    steps_per_episode:int=100000
    page_bytes:int=4096
    dram_ns:int=100
    nvm_ns:int=300
    ewma_alpha:float=0.05
    hotness_window:int=4096
    dram_capacity_pages:int=1000000
    lambda_migration:float=0.002
    lambda_writeamp:float=0.001
    lambda_mi_pressure:float=0.002
    ewma_mp_alpha:float=0.2

    # (Promote YN, # of Prefetch, Hotness threshold, Stride #)
    presets:List[Preset]=field(default_factory=lambda:[Preset(0,0,1.0,1),Preset(1,0,0.9,1)])

class TieredMemEnv:
    def __init__(self, cfg:EnvConfig, rng_seed:int=0):
        self.cfg=cfg
        self.rng=np.random.default_rng(rng_seed)
        self._gen=None
        self._obs_dim=5
        self.reset()

    @property
    def obs_dim(self): return self._obs_dim
    @property
    def action_dim(self): return len(self.cfg.presets)

    def set_generator(self, gen): self._gen=gen

    def reset(self):
        self.step_idx=0
        self.dram = OrderedDict()   # For LRU page management
        self.hotness={}; self._fault_ewma=0.0; self._stride_ewma=0.0
        self.last=[]; self.last_cap=self.cfg.hotness_window
        self.mig_pressure=0
        self._max_k = max(1 + p.prefetch for p in self.cfg.presets)
        self._prev = None
        return self._obs(0.0,0.0,0.0)

    def _obs(self, f, occ, mp): # State
        if self.last:
            ws_entropy=len(set(self.last))/len(self.last)
        else:
            ws_entropy=0.0
        stride_score=min(1.0, abs(self._stride_ewma)/128.0)

        # (Fault ewma, DRAM capa, Normalized Migration Pressure,
        # Page Dist Uniqueness(entropy), Normalized Stride # ewma)
        return np.array([f, occ, mp/self._max_k
                         , ws_entropy, stride_score], dtype=np.float32)

    def _qthr(self, q):
        if not self.hotness: return float('inf')
        v=np.fromiter(self.hotness.values(), dtype=np.float32)
        return float(np.quantile(v, q))

    def _dram_insert(self, page_id: int):
        self.dram[page_id] = None
        self.dram.move_to_end(page_id, last=True)   # LRU manner
        if len(self.dram) > self.cfg.dram_capacity_pages:
            self.dram.popitem(last=False)

    def step(self, action:int):
        assert self._gen is not None
        preset=self.cfg.presets[action]
        page,is_write=next(self._gen)

        # Update striding patern
        if self._prev is not None: self._stride_ewma=0.9*self._stride_ewma+0.1*(page-self._prev)
        self._prev=page

        # Calc page hotness
        a=self.cfg.ewma_alpha
        self.hotness[page]=(1-a)*self.hotness.get(page,0.0)+a*1.0
        self.last.append(page); 
        if len(self.last)>self.last_cap: self.last.pop(0)

        # Determine whether page is in DRAM (DRAM hit / miss)
        in_dram=page in self.dram
        latency=self.cfg.dram_ns if in_dram else self.cfg.nvm_ns
        fault=True; migrated=0
        if in_dram: self.dram.move_to_end(page, last=True); fault=False

        if (not in_dram) and preset.promote:
            if self.hotness.get(page,0.0) >= self._qthr(preset.hot_q):
                k=1+int(preset.prefetch); migrated=k;
                for i in range(k):  # Prefetch
                    self._dram_insert(page + i*preset.stride)

        # Reward
        avg_lat=latency*1e-9
        mig_bytes=migrated*self.cfg.page_bytes
        write_amp=max(0,migrated-1)*self.cfg.page_bytes
        self.mig_pressure=(1-self.cfg.ewma_mp_alpha)*self.mig_pressure + self.cfg.ewma_mp_alpha*migrated
        
        # Mean Latency | Migration bytes | Write Amp (due to additional prefetch) | Migraiton pressure
        r = -avg_lat \
            -self.cfg.lambda_migration*(mig_bytes/(1<<20)) \
            -self.cfg.lambda_writeamp*(write_amp/(1<<20)) \
            -self.cfg.lambda_mi_pressure*self.mig_pressure*(self.cfg.nvm_ns/4)

        self._fault_ewma=(1-self.cfg.ewma_alpha)*self._fault_ewma + self.cfg.ewma_alpha*(1.0 if fault else 0.0)
        self.step_idx+=1
        done=self.step_idx>=self.cfg.steps_per_episode
        obs=self._obs(self._fault_ewma, len(self.dram)/max(1,self.cfg.dram_capacity_pages), self.mig_pressure)
        info={"latency_ns":latency,"fault":fault,"migrated":migrated,"mig_pressure":self.mig_pressure}
        return obs, float(r), done, info
