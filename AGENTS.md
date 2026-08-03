# AGENTS.md — guide pour les outils d'édition automatique

Projet MSL (Machine Semantic Language). Python 3.11, PyTorch 2.13 (MPS sur Apple Silicon, CUDA si dispo).

## Commandes

- Installer (dépendances + package en editable) :
  `(.venv) pip install -e ".[dev]"`
- Lancer les tests : `(.venv) pytest`
- Tests avec couverture : `(.venv) pytest --cov=msl --cov-report=term-missing`
- Lint : `(.venv) ruff check src tests`
- Lint auto-fix : `(.venv) ruff check --fix src tests`
- Format : `(.venv) ruff format src tests`
- Typecheck : `(.venv) mypy src`
- Runner un entraînement : `(.venv) python -m msl.train.train_codec --config configs/mvp_h1_pq.yaml`

## Conventions

- Code en anglais (identifiants, commentaires), docstrings courtes.
- Pas de commentaires explicatifs sauf si la logique est non triviale.
- Tous les générateurs aléatoires doivent passer par `msl.utils.seeding` pour la reproductibilité.
- MS-1 est déterministe par graine : un état de test doit avoir une graine >= 2_000_000 (anti-fuite).
- Device par défaut : `mps` si dispo, sinon `cpu`. Pas de CUDA-only hardcodé.
- Les tests ne doivent pas dépendre du GPU ni du réseau.
