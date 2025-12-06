import argparse, os, csv, yaml, numpy as np, pandas as pd
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="runs/bandit_eval")
    ap.add_argument("--checkpoint", type=str, default="")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env_cfg, act_dim, tr = load_env_cfg(args.config)

    lat_csv = os.path.join(args.out, "latency_trace.csv")
    sum_csv = os.path.join(args.out, "summary.csv")
    act_csv = os.path.join(args.out, "action_dist.csv")

    with open(lat_csv, "w", newline="") as f_lat:
        wr_lat = csv.writer(f_lat)
        wr_lat.writerow(["episode","step","latency_ns","fault","migrated","mig_pressure","action"])

        for ep in range(args.episodes):

            # Set env
            rng = np.random.default_rng(args.seed + ep)
            env = TieredMemEnv(env_cfg, rng_seed=args.seed + ep)
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
            if args.checkpoint:
                agent.load(args.checkpoint)
            done=False; step=0

            # Do eval
            while not done:
                a = agent.act(obs)
                obs2, r, done, info = env.step(a)
                wr_lat.writerow([ep, step, info["latency_ns"], int(info["fault"]), info["migrated"],
                                 info.get("mig_pressure", 0.0), a])
                obs = obs2; step += 1

    # Summary
    df = pd.read_csv(lat_csv)
    rows = []
    for ep in sorted(df["episode"].unique()):
        sub = df[df["episode"]==ep]
        lat = sub["latency_ns"].values.astype(float)
        rows.append({
            "episode": int(ep),
            "mean_lat_ns": float(np.mean(lat)),
            "p95_lat_ns": float(np.quantile(lat, 0.95)),        # Usually same as worst latency
            "p99_lat_ns": float(np.quantile(lat, 0.99)),
            "migrated_pages": float(sub["migrated"].sum()),
            "faults": int(sub["fault"].sum()),
            "mig_pressure_mean": float(sub["mig_pressure"].mean() if "mig_pressure" in sub else 0.0),
        })
    pd.DataFrame(rows).to_csv(sum_csv, index=False)

    # Action Dist
    adf = df.groupby("action").size().reset_index(name="count")
    adf.to_csv(act_csv, index=False)

if __name__ == "__main__":
    main()
