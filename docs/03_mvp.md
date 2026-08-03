# MVP MSL — Spécification exécutable

Auteur : Kimi K3 (rôle)
Date : 31 juillet 2026
Objet : Spécifier le **MVP réalisable avec les ressources les plus modestes possibles** (brief §15). Conformément au brief §15, on ne code pas le système complet — on fournit l'architecture, le protocole expérimental, les métriques, les baselines, le budget et les critères go/no-go. Ce document constitue la **note d'architecture et le plan de reproductibilité** attendus en Phase 0 (brief §9).

---

## 1. Périmètre du MVP

Le MVP ne couvre **que les expériences H1 et H2** (cf. `02_experiences_falsification.md`). Tout le reste (H3 end-to-end, H4, H5, Transformer natif, standard inter-modèles) est hors périmètre et fait l'objet de phases ultérieures.

**Question unique du MVP** : *Un paquet sémantique discret peut-il coder un état complet de façon distribuée, et ce langage émergent est-il généralisable à un autre participant ?*

### Réponses attendues (livrables)
- Une courbe `L(s)` vs `k` (H1).
- Une matrice de transfert inter-codecs (H2).
- Un verdict go/no-go documenté.

### Non-livrables
- Aucun Transformer MSL (Phase 3).
- Aucun texte réel (Phase 2).
- Aucun canal littéral optimisé (H5, Phase 2).
- Aucune standardisation binaire (Phase 4).

## 2. Principe de sobriété

Le MVP est conçu pour **tenir sur 1 GPU consommateur (24 Go VRAM, type RTX 4090)** en ~3 semaines calendaires. Toute décision d'architecture est prise sous cette contrainte. Les sur-dimensionnements possibles sont documentés en §10 (scaling).

---

## 3. Architecture du MVP

### 3.1 Schéma général

```
                  Encoder E                Quantizer Q            Decoder D
texte(s) ──► [small transformer] ──► z ──► [PQ/RVQ/FSQ] ──► c[1..n] ──► [small transformer] ──► ŝ
                                                                                    │
                                                                                    ▼
                                                                       Tâches(QA, impl, contr, ...)
                                                                                    │
                                                                                    ▼
                                                                                  Vérifier V
```

L'encodeur et le décodeur sont **petits** (≤ 30 M chacun). Le cœur MSL (Transformer T) **n'est pas dans le MVP** : on teste le codec, pas le Transformer natif. Les paquets `c[1..n]` sont donc consommés directement par D, sans étape de prédiction autorégressive. C'est conforme au brief §6 : « La première réalisation du projet n'est pas le Transformer MSL. C'est le codec bidirectionnel. »

### 3.2 Modules

| Module | Rôle | Implémentation MVP |
|---|---|---|
| **E** (encodeur) | texte(s) → z (continu) | Transformer 6 layers, d_model=256, 8 heads, ~20 M params. Entrée = tokens BPE (V=4k) du texte MS-1. Sortie = z ∈ R^d_z avec d_z = 64. |
| **Q** (quantizer) | z → c[1..n] | Un des 3 quantizers (§3.3). STOP via slot dédié. |
| **D** (décodeur) | c[1..n] + (langue, style) → ŝ | Transformer 6 layers, d_model=256, 8 heads, ~20 M params. |
| **V** (vérificateur) | Évalue les tâches sur ŝ vs s. | Non paramétrique : règle exécutoire MS-1. |

### 3.3 Quantizer (3 variantes à comparer)

Le brief §5.2 impose de comparer **PQ, RVQ et FSQ**. Spécifications minimales :

- **PQ (Product Quantization)** : `z ∈ R^64` découpé en `B=8` sous-vecteurs de dim 8, chacun quantifié dans un codebook de `V=1024`. 8 codes en parallèle, prédits par 8 têtes indépendantes. 80 bits/paquet.
- **RVQ (Residual VQ)** : `B=8` codebooks successifs, chacun appris sur le résidu du précédent. 80 bits/paquet.
- **FSQ (Finite Scalar Quantization)** : `B=8` dims, chaque dim quantifiée à `L=5` niveaux → 8 codes. ~18 bits/paquet (config plus légère, utile comme point bas).

Toutes les configurations sont fixées pour que le budget total par paquet soit comparable, **sauf** la config FSQ qui sert volontairement de borne basse pour vérifier que les gains ne sont pas dûs seulement à la largeur du codebook.

### 3.4 Canal littéral (désactivé dans le MVP)

Pour isoler H1 et H2, le canal littéral est **désactivé** : aucune valeur exacte ne passe par ce canal. Les littéraux sont *supprimés* des états MS-1 du MVP (les noms, nombres et dates sont remplacés par des symboles génériques dans le texte d'entrée). On testera H5 (canal littéral) en Phase 2.

### 3.5 Format d'un paquet (figure structurelle)

```
packet = (q1, q2, q3, q4, q5, q6, q7, q8)   # 8 codes PQ
embedding = project(concat(emb(qi))) ∈ R^256
```

L'embedding du paquet est une composition apprise (concat + projection linéaire), conformément au brief §5.2 (« la somme, la concaténation projetée ou une composition apprise »).

---

## 4. Données — Monde synthétique MS-1

### 4.1 Générateur

Implémentation : `src/data/ms1.py`. Caractéristiques (cf. `02_experiences_falsification.md` §0) :

- États = graphes attribués (entités, relations, attributs, événements, modalités).
- Génération **infinie** avec graine, splits train/val/test déterministes par plages de graines.
- Vérité de terrain **calculée** par exécution d'un interprète MS-1 (pas d'annotation).
- Tâches : QA, implication, contradiction, temps, composition, (littéraux retirés pour le MVP).

### 4.2 Splits

- **train** : graines 0–999 999.
- **val** : graines 1 000 000–1 009 999.
- **test** : graines 2 000 000–2 009 999 (splits inédits garantis).
- **test compositionnel** (pour H6 bonus) : combinaisons de relations **jamais vues** en train, générées par tirage conditionnel.

### 4.3 Formulation textuelle

Chaque état est converti en texte par un **générateur paramétrique** (templates aléatoires avec variations syntaxiques, paraphrases, ordres de présentation). On prévoit :

- 4 paraphrases par état en moyenne (variations lexicales, syntaxiques).
- 2 langues : français et anglais (MVP bilingue minimal, pour préparer l'invariance de formulation).
- 1 vue structurée (JSON-like).

Cela donne 4 × 2 + 1 = 9 vues par état, utilisées en `L_alignement_multivue` (brief §6.2).

### 4.4 Taille

- 10 M états entraînement (graines 0–999 999), générés à la volée (pas de stockage).
- 10 k états validation.
- 10 k états test.
- 1 k états test compositionnel.

---

## 5. Protocole expérimental

### 5.1 Phase 1A — H1 (capacité du paquet)

1. Entraîner (E, Q, D) sur MS-1, sans canal littéral, objectif `L_total` (§6).
2. Pour k ∈ {2, 4, 8, 16, 32, 64}, générer 1000 états inédits par palier, encoder-décoder, mesurer `L(s)` minimal garantissant récupération parfaite (tâches à 100 %).
3. Répéter pour les 3 quantizers (PQ, RVQ, FSQ) et 3 configs `(B, V)` (§6).
4. Régression `L(s) ~ a·k + b`, comparer `a / a_text`.
5. **Test H1** (cf. `02_experiences_falsification.md` §H1).

### 5.2 Phase 1B — H2 (standard non privé)

1. Entraîner N = 8 émetteurs indépendants (seeds différentes).
2. Pour chaque paire (A, B), A ≠ B :
   - intra-codec : récepteur naïf sur A, test intra.
   - inter-codec : récepteur déjà entraîné sur A, test sur B en zero-shot, puis fine-tuning léger (5 % du budget initial).
   - chance : récepteur fraîchement initialisé sur B (avant entraînement).
3. Répéter avec condition **multi-agent** (émetteurs + récepteurs entraînés ensemble via jeu de référence, brief §7.1) vs **mono-agent** (entraînement isolé) pour isoler l'effet multi-agent.
4. **Test H2** (cf. `02_experiences_falsification.md` §H2).

### 5.3 Hyperparamètres communs

- Optimiseur : AdamW, lr = 3e-4, warmup 1k steps, cosine decay.
- Batch : 256 états (étalement sur gradient accumulation si besoin).
- Steps : 200 k par run (suffisant d'après Coconut / LCM à cette échelle).
- Seeds : 3 par configuration (sauf H2 qui en a 8 par design).

---

## 6. Fonctions de perte

Le brief §6.2 impose :

```
L_total = L_semantique
        + alpha * L_reconstruction
        + beta  * L_alignement_multivue
        + gamma * L_taches
        + delta * L_compositionnalite
        + lambda * cout_bits
        + mu    * cout_calcul
```

### Spécifications MVP

| Terme | Définition | Poids MVP |
|---|---|---|
| `L_semantique` | Cross-entropy sur les tâches MS-1 (QA, impl, contr, temps). | 1.0 |
| `L_reconstruction` | Cross-entropy token sur les 9 vues textuelles (auxiliaire). | α = 0.3 |
| `L_alignement_multivue` | Cosine entre `z(vue_i)` et `z(vue_j)` pour les 9 vues d'un même état. | β = 0.2 |
| `L_taches` | Pénalité sur erreurs aux tâches exécutées. | γ = 0.5 |
| `L_compositionnalite` | Régularisation PosDis (Chaabouni et al.) sur les codes. | δ = 0.05 |
| `cout_bits` | Pénalité linéaire sur `n_paquets × B × log2(V)`. | λ = 1e-4 |
| `cout_calcul` | Estimation FLOPS (mesuré, pas pénalité dure dans le MVP). | μ = 0 (MVP) |

Les poids α, β, γ, δ sont **des points de départ** à explorer sur la courbe débit-distorsion (brief §6.2). On les balaye en grille restreinte (3 valeurs chacun, sobriété).

---

## 7. Métriques

### 7.1 Métriques primaires (pour les critères go/no-go)

| Métrique | Définition | Utilisée pour |
|---|---|---|
| `L(s)` | Longueur MSL minimale garantissant 100 % de réussite aux tâches sur `s`. | H1 |
| `a / a_text` | Pente relative de `L(s)` vs complexité. | H1 |
| `T_intra`, `T_inter`, `T_chance` | Taux de réussite aux tâches (intra-codec, inter-codec, chance). | H2 |
| `PosDis` | Positional disentanglement. | H2 |
| Codebook usage | Perplexité des codes, % codes actifs. | Anti-faux-négatif H1 |

### 7.2 Métriques secondaires (descriptives, pas go/no-go)

- Courbe débit-distorsion : qualité vs bits transmis.
- BAscore (alignment de codebooks).
- Engagement des codes (% de codes utilisés > 1‰).
- Loss de chaque terme de `L_total` au cours de l'entraînement (pour audit).
- FLOPS mesurés via `fvcore` (descriptif, pas critère — H3 est hors MVP).

### 7.3 Reproductibilité

- Toutes les runs enregistrées via `wandb` ou `tensorboard`.
- Configuration YAML versionnée dans `configs/`.
- Seeds fixées (random, numpy, torch, CUDA).
- Générateur MS-1 déterministe et versionné (`ms1.__version__`).
- Tout résultat publié avec : seed, config, commit git, version PyTorch, version CUDA.

---

## 8. Baselines

Le brief §10 impose « texte tokenisé, latent continu, latent discret ». Spécifications MVP :

| Baseline | Description | Rôle |
|---|---|---|
| **B-text** | Transformer texte (~30 M) entraîné sur la formulation textuelle de MS-1. Reconstruit l'état puis exécute les tâches. | Renvoie `a_text` pour H1. |
| **B-latent-cont** | Encodeur continu → z (pas de Q) → décodeur continu (type Coconut). | Réf. « latent continu » du brief. |
| **B-one-hot** | Un code one-hot par atome sémantique (pas de compression). | Borne inférieure « pas de distribution » pour H1. |
| **B-text-baseline** | Bon tokenizer BPE standard (V=4k), pas de quantizer. | Contrôle que la formulation MS-1 n'est pas artificiellement verbeuse. |

**MSL** est comparé à ces 4 baselines sur **toutes les métriques**. Si MSL ne bat pas B-one-hot sur H1, le verdict est « pas de distribution ». Si MSL ne bat pas B-text-baseline, c'est la formulation MS-1 qui est en cause, pas le codec.

---

## 9. Budget

### 9.1 Calcul

Postulé : 1 GPU RTX 4090 (24 Go), FP16/BF16, batch via gradient accumulation.

| Expérience | Runs | Steps/run | GPU-heures |
|---|---|---|---|
| Phase 1A (H1) — 3 quantizers × 3 configs × 3 seeds = 27 | 27 | 200 k | ~270 h |
| Phase 1A — 4 baselines × 3 seeds | 12 | 200 k | ~120 h |
| Phase 1B (H2) — 8 émetteurs + ~56 paires × 1 test + 3 seeds multi-agent | ~80 | 100 k | ~530 h |
| Évaluation et debug | — | — | ~150 h |
| **Total** | | | **~1070 h** |

Soit ≈ **45 jours calendaires sur 1 GPU 24 h/24**, ou **3 semaines sur 2 GPUs**. Pour le MVP « le plus modeste possible », on accepte jusqu'à 4 semaines sur 1 GPU en séquentiel.

### 9.2 Stockage

- Checkpoints : 5 Go × 80 runs × 3 seeds = 1.2 To si on garde tout. **Stratégie** : garder seulement le dernier checkpoint + le meilleur (val loss). → 200 Go.
- Logs W&B/TB : ~50 Go.
- Données MS-1 : générées à la volée, ~0 stockage.

### 9.3 Coût financier (estimation, hors hardware)

- AWS p3.2xlarge (V100 16 Go) ≈ 3 USD/h. 1070 h ≈ 3200 USD.
- Lambda Labs A10 (24 Go) ≈ 0.75 USD/h. 1070 h ≈ 800 USD.
- Maison (RTX 4090 déjà possédée) : 0 USD + électricité (~30 EUR).

Le MVP est jouable **à domicile sur GPU consommateur**. C'est la cible de sobriété du brief §15.

---

## 10. Critères go/no-go

Le brief §11.1 fixe des cibles chiffrées mais dit lui-même que ce sont des « hypothèses de travail à recalibrer après les premières courbes débit-distorsion ». Les critères ci-dessous sont donc **calibrés sur H1 et H2 uniquement** (les deux hypothèses testées par le MVP), pas sur les cibles de §11.1 qui relèvent de phases ultérieures.

### 10.1 Critère H1 (capacité du paquet)

- **GO** si `a / a_text ≤ 0.4` sur k ∈ [4, 32] (intervalle de confiance 95 %) ET codebook usage ≥ 80 %.
- **NO-GO** si `a / a_text ≥ 0.8` sur la même plage.
- **Zone grise** : investiguer (collapse de codebook ? mauvais quantifier ? bad split ?) avant tout verdict.

### 10.2 Critère H2 (standard non privé)

- **GO** si `T_inter ≥ 0.7 · T_intra` en zero-shot, **ou** `T_inter ≥ T_intra − 2 pts` après fine-tuning léger.
- **NO-GO** si `T_inter ≤ T_chance + 5 pts` dans les deux conditions.
- **Zone grise** : intermédiaire → investiguer ( PosDis faible ? codecs trop similaires par hasard ?).

### 10.3 Critère global (après Phase 1)

| H1 | H2 | Décision |
|---|---|---|
| GO | GO | **GO Phase 2** (codec sémantique bidirectionnel + texte multilingue + canal littéral + H5). |
| GO | NO-GO | **NO-GO ou reformulation** : le codec marche mais n'est pas standard. Option : standardiser par distillation (imposer un codec de référence). |
| NO-GO | GO | **NO-GO ou reformulation** : le langage émerge et est standard, mais les paquets sont trop longs. Option : élargir les codebooks, ou repenser l'atome (paquet = séquence courte, pas unité). |
| NO-GO | NO-GO | **NO-GO** : le concept même de paquet-état-complet-opaque n'est pas viable à cette échelle. Documenter et arrêter. |

### 10.4 Critères de qualité secondaires (non bloquants)

- Si `B_latent-cont` bat MSL sur H1 par > 20 %, le goulot discret coûte trop cher par rapport au latent continu. Pas un no-go, mais signal à investiguer pour Phase 2.
- Si `B-one-hot` bat MSL sur H1, **red flag** : le système n'a rien appris, on vérifie le pipeline.

---

## 11. Structure du dépôt

```
msl_project/
├── Brief_projet_MSL_Kimi_K3.docx     # Brief original (ne pas modifier)
├── README.md                          # Orientation du dépôt
├── docs/
│   ├── 01_audit_critique.md            # Audit scientifique
│   ├── 02_experiences_falsification.md # 5 expériences
│   └── 03_mvp.md                       # Ce document
├── configs/                            # YAML versionnés
│   ├── mvp_h1_pq.yaml
│   ├── mvp_h1_rvq.yaml
│   ├── mvp_h1_fsq.yaml
│   └── mvp_h2.yaml
├── src/
│   ├── data/
│   │   └── ms1.py                       # Générateur MS-1 (à écrire en Phase 1)
│   ├── models/
│   │   ├── encoder.py
│   │   ├── quantizer.py                # PQ, RVQ, FSQ
│   │   └── decoder.py
│   ├── train/
│   │   ├── train_codec.py              # Phase 1A
│   │   └── train_emitter_receiver.py   # Phase 1B
│   ├── eval/
│   │   ├── h1_capacity.py
│   │   ├── h2_transfer.py
│   │   └── metrics.py
│   └── utils/
│       ├── flops.py                    # fvcore wrapper
│       └── seeding.py
├── tests/                              # Pytest, à écrire en Phase 1
│   ├── test_ms1.py
│   ├── test_quantizers.py
│   └── test_codec.py
└── data/
    └── notebooks/                      # Analyses post-run
        └── plots.ipynb
```

**Remarque** : ce dépôt ne contient **pas encore de code** — conformément au brief §15 (« Ne code pas encore le système complet »). Le lancement du code se fera après validation de ce document par le porteur de projet.

---

## 12. Risques MVP-spécifiques et parades

| Risque MVP | Manifestation | Parade |
|---|---|---|
| Collapse de codebook | < 20 % codes actifs → H1 fausse négative. | EMA codebook, restart des codes morts, mesurer usage à chaque eval. |
| MS-1 trop simple | Tout marche, pas informatif. | Inclure la borne `k=64` (8× la complexité nominale). Si H1 passe même à k=64, on étendra. |
| MS-1 trop dur | Rien ne marche. | Commencer par k ∈ {2, 4, 8}, ne monter que si ça passe. |
| 8 émetteurs qui convergent identiques | H2 faux positif (codes privés mais identiques). | Mesurer la diversité des codes (Jaccard entre codebooks), exiger ≥ 30 % de différence. |
| Bug de split | Fuite train→test, fausse performance. | Audit : un état de test doit avoir une graine ≥ 2 000 000, vérification automatisée. |
| Sur-apprentissage du décodeur | D apprend à deviner sans c. | Périodiquement, remplacer c par du bruit et mesurer la chute de qualité (doit être ≥ 50 %). |

---

## 13. Calendrier indicatif (3 semaines, 1 GPU)

| Semaine | Activité | Livrable |
|---|---|---|
| S1 | Implémentation MS-1 + encodeur/décodeur/quantizers + tests unitaires. | Code compilable, baselines qui tournent. |
| S2 | Runs Phase 1A (H1) + premières baselines + courbes préliminaires. | Courbes `L(s)` vs k, verdict H1 préliminaire. |
| S3 | Runs Phase 1B (H2) + analyse + rapport. | Matrice de transfert, rapport go/no-go. |

---

## 14. Prochaines étapes (après ce document)

1. **Validation par le porteur** : ce document doit être validé par Eric Houzelle avant tout codage.
2. Si validé : création du `AGENTS.md` avec commandes de lint/typecheck/test.
3. Implémentation de MS-1 et des tests unitaires (S1).
4. Lancement des runs Phase 1A et 1B.

Ce document ne spécifie **pas** le code : conformément au brief §15, l'architecture et le protocole sont posés, le code viendra après validation.
