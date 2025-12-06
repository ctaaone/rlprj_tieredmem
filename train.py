import argparse, yaml, csv, os, numpy as np
from envs.tiered_mem_env import TieredMemEnv, EnvConfig, Preset
from traces.synth import phased_memmap_like
from buffers.replay import ReplayBuffer
from agents.dqn import DQNAgent, DQNConfig

def load_cfg(path):
    with open(path,"r") as f: raw=yaml.safe_load(f)
    p=[Preset(int(e["promote"]),int(e["prefetch"]),float(e["hot_q"]),int(e["stride"])) for e in raw["env"]["presets"]]
    e=raw["env"]
    env_cfg=EnvConfig(
        steps_per_episode=int(e["steps_per_episode"]), page_bytes=int(e["page_bytes"]),
        dram_ns=int(e["dram_ns"]), nvm_ns=int(e["nvm_ns"]), ewma_alpha=float(e["ewma_alpha"]),
        hotness_window=int(e["hotness_window"]), dram_capacity_pages=int(e["dram_capacity_pages"]),
        lambda_migration=float(e["lambda_migration"]), lambda_writeamp=float(e["lambda_writeamp"]),
        lambda_mi_pressure=float(e["lambda_mi_pressure"]), ewma_mp_alpha=float(e["ewma_mp_alpha"]),
        presets=p
    )
    a=raw["agent"]
    agent_cfg=DQNConfig(obs_dim=5, act_dim=len(p), gamma=float(a["gamma"]), lr=float(a["lr"]),
                        double=bool(a["double"]), dueling=bool(a["dueling"]), target_tau=float(a["target_update_tau"]))
    extras={"batch_size":int(a["batch_size"]), "replay_size":int(a["replay_size"]), "warmup_steps":int(a["warmup_steps"]),
            "eps_start":float(a["eps_start"]), "eps_end":float(a["eps_end"]), "eps_decay_steps":int(a["eps_decay_steps"]),
            "hidden":list(a["hidden"]), "trace": raw["trace"]}
    return env_cfg, agent_cfg, extras

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--steps",type=int,default=100000)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",type=str,default="runs/")
    args=ap.parse_args()

    # Set env
    os.makedirs(args.out, exist_ok=True)
    env_cfg, agent_cfg, ex=load_cfg(args.config)
    rng=np.random.default_rng(args.seed)
    env=TieredMemEnv(env_cfg, rng_seed=args.seed)
    tr=ex["trace"]
    gen=phased_memmap_like(rng, footprint_pages=int(tr["footprint_pages"]),
                           phase_seconds=int(tr["phase_seconds"]),
                           seq_jump_pages=int(tr["seq_jump_pages"]),
                           stride_k_pages=int(tr["stride_k_pages"]))
    env.set_generator(gen)

    # Set RL agent
    import torch
    torch.manual_seed(args.seed+1206)
    agent=DQNAgent(agent_cfg, hidden=tuple(ex["hidden"]))
    rb=ReplayBuffer(obs_dim=agent_cfg.obs_dim, seed=args.seed, capacity=int(ex["replay_size"]))
    eps=ex["eps_start"]; decay=(ex["eps_start"]-ex["eps_end"])/max(1,ex["eps_decay_steps"])
    obs=env.reset(); step=0

    # Do train
    with open(os.path.join(args.out,"train_log.csv"),"w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["step","reward","loss","eps","latency_ns","fault","migrated","mig_pressure"])
        while step<args.steps:
            a=agent.act_eps(obs,eps)            # Exploration
            obs2,r,done,info=env.step(a)
            rb.add(obs,a,r,obs2,float(done)); obs=obs2
            if eps>ex["eps_end"]: eps-=decay    # Decaying epsilon
            loss=0.0
            if rb.size>=ex["warmup_steps"]:
                batch=rb.sample(ex["batch_size"]); loss=agent.update(batch, rb)
            wr.writerow([step,r,loss,eps,info["latency_ns"],int(info["fault"]),info["migrated"],info["mig_pressure"]])
            step+=1
            if done: obs=env.reset()
    agent.save(os.path.join(args.out,"checkpoint.pt"))
    print("done:", args.out)

if __name__=="__main__": main()
