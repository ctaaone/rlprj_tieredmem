import argparse, os, yaml, numpy as np, pandas as pd
from envs.tiered_mem_env import TieredMemEnv, EnvConfig, Preset
from traces.synth import phased_memmap_like
from agents.dqn import DQNAgent, DQNConfig

def load_cfg(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    P = [Preset(int(e["promote"]), int(e["prefetch"]), float(e["hot_q"]), int(e.get("stride",1)))
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
        presets=P,
    )
    a = raw["agent"]
    agent_cfg = DQNConfig(
        obs_dim=5,
        act_dim=len(P),
        gamma=float(a["gamma"]),
        lr=float(a["lr"]),
        double=bool(a["double"]),
        dueling=bool(a["dueling"]),
        target_tau=float(a["target_update_tau"]),
    )
    hidden = list(a["hidden"])
    t = raw["trace"]
    trace = dict(footprint_pages=int(t["footprint_pages"]), phase_seconds=int(t["phase_seconds"]),
                 seq_jump_pages=int(t["seq_jump_pages"]), stride_k_pages=int(t["stride_k_pages"]))
    return env_cfg, agent_cfg, hidden, trace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="runs/eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env_cfg, agent_cfg, hidden, tr = load_cfg(args.config)

    lat_rows = []
    act_rows = []

    for ep in range(args.episodes):
        # Set env
        rng = np.random.default_rng(args.seed + ep)
        env = TieredMemEnv(env_cfg, rng_seed=args.seed + ep)
        gen = phased_memmap_like(
            rng=rng,
            footprint_pages=int(tr["footprint_pages"]),
            phase_seconds=int(tr["phase_seconds"]),
            seq_jump_pages=int(tr["seq_jump_pages"]),
            stride_k_pages=int(tr["stride_k_pages"]),
        )
        env.set_generator(gen)

        # Set agent
        agent = DQNAgent(agent_cfg, hidden=tuple(hidden))
        agent.load(args.checkpoint)

        obs = env.reset()
        done = False; step = 0

        # Do eval
        while not done:
            a = agent.act_eps(obs, eps=0.0)
            obs2, r, done, info = env.step(a)
            lat_rows.append([ep, step, info["latency_ns"], int(info["fault"]),
                             info["migrated"], info.get("mig_pressure", 0.0), a])
            act_rows.append([a])
            obs = obs2; step += 1

    df = pd.DataFrame(lat_rows, columns=["episode","step","latency_ns","fault","migrated","mig_pressure","action"])
    df.to_csv(os.path.join(args.out, "latency_trace.csv"), index=False)

    # Summary
    rows = []
    for ep in sorted(df["episode"].unique()):
        sub = df[df["episode"] == ep]
        lat = sub["latency_ns"].to_numpy(dtype=float)
        rows.append({
            "episode": int(ep),
            "mean_lat_ns": float(np.mean(lat)),
            "p95_lat_ns": float(np.quantile(lat, 0.95)),        # Same as worst latency, nomally
            "p99_lat_ns": float(np.quantile(lat, 0.99)),        #
            "migrated_pages": float(sub["migrated"].sum()),
            "faults": int(sub["fault"].sum()),
            "mig_pressure_mean": float(sub["mig_pressure"].mean()),
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # Action dist
    adf = pd.DataFrame(act_rows, columns=["action"]).value_counts().reset_index(name="count")
    adf.to_csv(os.path.join(args.out, "action_dist.csv"), index=False)

if __name__ == "__main__":
    main()
