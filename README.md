# Tiered Memory 환경에서의 디바이스별 Cache Replacement RL Policy 비교 분석
시뮬레이션 된 환경에서 CXL-memory / SSD 디바이스별 학습된 cache replacement policy의 차이를 비교 분석한다.

60조 / 이승재

## 보고서
REPORT_team60.pdf

## 모델 경로
- models/ssd_pt.pt
- models/cxld_pt.pt


# 실행 방법
## Dependency
pip install numpy pandas pyyaml torch

## Train & Eval (seeds=[0, 1, 2, 3, 4])
python run.py --seeds 0 1 2 3 4 --steps 100000 --episodes 5 --out_root runs

## Train only
### CXL-mem
python train.py --config configs/cxld.yaml --steps 100000 --seed 0 --out runs/cxld_s0
### SSD
python train.py --config configs/ssd.yaml --steps 100000 --seed 0 --out runs/ssd_s0

## Eval only
학습 모델은 학습 디렉토리 안에 "checkpoint.pt"로 저장됨
### CXL-mem
python eval_policy.py --config configs/cxld.yaml --checkpoint "PATH_TO_CKPT" --episodes 5 --seed 0 --out runs/cxld_s0_eval
### SSD
python eval_policy.py --config configs/ssd.yaml --checkpoint "PATH_TO_CKPT" --episodes 5 --seed 0 --out runs/ssd_s0_eval


## MAB algorithm (비교용)
## Train
python train_bandit.py --config configs/cxld.yaml --alpha 0.5 --l2 1.0 --steps 100000 --seed 0 --out runs/mab_cxld_s0
python train_bandit.py --config configs/ssd.yaml --alpha 0.5 --l2 1.0 --steps 100000 --seed 0 --out runs/mab_ssd_s0

## Eval
python eval_bandit.py --config configs/cxld.yaml \
  --alpha 0.0 \
  --checkpoint runs/mab_cxld_s0/bandit_ckpt.npz \
  --episodes 5 --seed 0 --out runs/mab_cxld_s0_eval

python eval_bandit.py --config configs/ssd.yaml \
--alpha 0.0 \
--checkpoint runs/mab_ssd_s0/bandit_ckpt.npz \
--episodes 5 --seed 0 --out runs/mab_ssd_s0_eval