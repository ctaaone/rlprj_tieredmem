import argparse, os, csv, yaml, numpy as np
from envs.tiered_mem_env import TieredMemEnv, EnvConfig, Preset
from traces.synth import phased_memmap_like
from agents.bandit import LinUCBBandit, LinUCBConfig

def load_env_cfg(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    p = [Preset(int(e["promote"]), int(e["prefetch"]), float(e["hot_q"]), int(e.get("stride", 1)))
         for e in raw["env"]["presets"]]
    e = raw["env"]
    env_cfg = EnvConfig(
        steps_per_episode=int(e["steps_per_episode"]),
        page_bytes=int(e["page_bytes"]),
        dram_ns=int(e["dram_ns"]),
        nvm_ns=int(e["nvm_ns"]),
        ewma_alpha=float(e["ewma_alpha"]),
        hotness_window=int(e["hotness_window"]),
        dram_capacity_pages=int(e["dram_capacity_pages"]),
        lambda_migration=float(e["lambda_migration"]),
        lambda_writeamp=float(e["lambda_writeamp"]),
        lambda_mi_pressure=float(e["lambda_mi_pressure"]),
        ewma_mp_alpha=float(e["ewma_mp_alpha"]),
        presets=p
    )
    return env_cfg, len(p), raw["trace"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/bandit")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    # Set env
    os.makedirs(args.out, exist_ok=True)
    env_cfg, act_dim, tr = load_env_cfg(args.config)
    rng = np.random.default_rng(args.seed)
    env = TieredMemEnv(env_cfg, rng_seed=args.seed)
    gen = phased_memmap_like(rng=rng, footprint_pages=int(tr["footprint_pages"]), phase_seconds=int(tr["phase_seconds"]),
                             seq_jump_pages=int(tr["seq_jump_pages"]), stride_k_pages=int(tr["stride_k_pages"]))
    env.set_generator(gen)
    obs = env.reset()

    # Set RL agent
    try:
        d = env.obs_dim
    except Exception:
        d = len(obs)
    agent = LinUCBBandit(LinUCBConfig(n_actions=act_dim, d=d, alpha=args.alpha, l2=args.l2))

    train_csv = os.path.join(args.out, "train.csv")
    with open(train_csv, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["step","reward","action"])
        step = 0; done = False
        # Do train
        while not done and step < args.steps:
            a = agent.act(obs)
            obs2, r, done, info = env.step(a)
            agent.update(obs, a, r)
            wr.writerow([step, r, a])
            obs = obs2; step += 1

    ckpt = os.path.join(args.out, "bandit_ckpt.npz")
    try:
        agent.save(ckpt)
    except Exception:
        pass

if __name__ == "__main__":
    main()
