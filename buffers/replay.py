import numpy as np
class ReplayBuffer:
    def __init__(self, obs_dim, seed, capacity=50000):
        self.capacity=capacity
        self.obs=np.zeros((capacity,obs_dim),np.float32)
        self.act=np.zeros((capacity,),np.int64)
        self.rew=np.zeros((capacity,),np.float32)
        self.nxt=np.zeros((capacity,obs_dim),np.float32)
        self.done=np.zeros((capacity,),np.float32)
        self.ptr=0; self.size=0
        self.rng = np.random.default_rng(seed)
    def add(self,o,a,r,o2,d):
        i=self.ptr%self.capacity
        self.obs[i]=o; self.act[i]=a; self.rew[i]=r; self.nxt[i]=o2; self.done[i]=d
        self.ptr+=1; self.size=min(self.size+1,self.capacity)
    def sample(self,batch):
        idx=self.rng.integers(0,self.size,size=batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.nxt[idx], self.done[idx]
