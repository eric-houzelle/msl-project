# Installation et déploiement sur Debian 13 + A10

## 1. Pré-requis système

```bash
# Debian 13
sudo apt update && sudo apt install -y python3.11 python3.11-venv git curl

# CUDA 12.x (si pas déjà installé)
# Suivre la doc NVIDIA : https://docs.nvidia.com/cuda/
# Vérifier : nvidia-smi doit afficher la A10
```

## 2. Cloner le projet

```bash
git clone <repo> msl_project
cd msl_project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 3. Vérifier l'environnement

```bash
python -c "import torch; print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
```

Doit afficher : `torch 2.x, cuda True, device NVIDIA A10`

## 4. Télécharger le dataset (100k phrases)

```bash
python -u -m msl.data.download_hf --size 100000 --out runs/big_corpus.pt
```

## 5. Entraîner le décodeur (20k steps, ~2-4h sur A10)

```bash
python -u -m msl.train.train_text_decoder --steps 20000 --seed 0 --lr 3e-5
```

## 6. Tester le round-trip

```bash
python -u -m msl.eval.test_roundtrip_realtext
```

## 7. Tests H2

```bash
python -u -m msl.eval.test_h2
```
