import argparse, subprocess, pathlib

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out_root", type=str, default="runs")
    args = ap.parse_args()

    pathlib.Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for name, cfg in [("ssd", "configs/ssd.yaml"), ("cxld", "configs/cxld.yaml")]:
        for s in args.seeds:
            tr_out = f"{args.out_root}/{name}_s{s}"
            ev_out = f"{args.out_root}/{name}_s{s}_eval"
            pathlib.Path(tr_out).mkdir(parents=True, exist_ok=True)
            pathlib.Path(ev_out).mkdir(parents=True, exist_ok=True)

            run(["python", "train.py", "--config", cfg, "--steps", str(args.steps),
                 "--seed", str(s), "--out", tr_out])

            run(["python", "eval_policy.py", "--config", cfg,
                 "--checkpoint", f"{tr_out}/checkpoint.pt",
                 "--episodes", str(args.episodes), "--seed", str(s), "--out", ev_out])

if __name__ == "__main__":
    main()
