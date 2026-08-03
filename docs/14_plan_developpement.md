# Plan de développement MSL — Version 1.0

Auteur : Kimi K3 (rôle) + Eric Houzelle
Date : 3 août 2026
Objet : Plan séquentiel pour passer du prototype au système complet. Chaque étape doit être terminée et mesurée avant de passer à la suivante.

---

## Phase A — Pipeline complet avec MiniLM (en cours)

### A.1. Décodeur MSL → texte
- **Ce qu'on fait** : GPT-2 (124M, 2 dernières layers débloquées) apprend à reconstruire du texte depuis les paquets MSL.
- **Encodeur** : all-MiniLM-L6-v2 (22M params, 384-dim)
- **Quantizer** : PQ 48 codebooks × 256 = 384 bits
- **Données** : 50k phrases Tatoeba (anglais)
- **Entraînement** : 20k steps, batch 16, lr 3e-5, sur A10
- **Statut** : **En cours sur A10**
- **Critère de succès** : 4-5/5 paraphrases fidèles (avec quantization réelle)

### A.2. Corpus MSL réel
- **Ce qu'on fait** : Convertir les 50k phrases en paquets MSL avec le codec figé.
- **Commande** : `python -u -m msl.data.build_msl_corpus --codec runs/text_decoder_quant_0.pt --size 50000 --out runs/msl_corpus_real.pt`
- **Statut** : Prêt à lancer après A.1
- **Critère** : 50k paquets, diversité > 90% (pas de collapse)

### A.3. LLM natif MSL
- **Ce qu'on fait** : Entraîner un Transformer (6.8M params) uniquement sur les paquets MSL — jamais de texte.
- **Commande** : `python -u -m msl.train.train_native_lm --corpus runs/msl_corpus_real.pt --steps 20000`
- **Statut** : Prêt à lancer après A.2
- **Critère** : > 85% codes corrects, > 40% paquets exacts

### A.4. Mesure du gain économique
- **Ce qu'on fait** : Comparer le LLM MSL natif vs un LLM texte jumeau (même taille, même données).
- **Métriques** : FLOPS, mémoire, latence, qualité
- **Commande** : `python -u -m msl.eval.end_to_end`
- **Statut** : Prêt à lancer après A.3
- **Critère** : Gain > 3× plus rapide end-to-end (vs 6× sur synthétique)

### A.5. Test H2 sur vrai texte
- **Ce qu'on fait** : Refaire le test de standard inter-modèles sur le corpus réel.
- **Commande** : `python -u -m msl.eval.test_h2`
- **Statut** : Prêt à lancer après A.3
- **Critère** : > 80% adoption (vs 87% sur synthétique)

---

## Phase B — BGE-M3 (amelioration de l'encodeur)

### B.1. Corpus BGE-M3
- **Ce qu'on fait** : Re-encoder les 50k phrases avec BGE-M3 (568M params, 1024-dim, multilingue).
- **Encodeur** : BAAI/bge-m3 (568M params, 1024-dim, 100+ langues)
- **Quantizer** : PQ 64 codebooks × 256 = 512 bits (plus de capacité)
- **Commande** : `python -u -m msl.data.download_hf_bge --size 100000 --out runs/bge_corpus.pt`
- **Statut** : Code prêt, à lancer après Phase A
- **Critère** : Gap sémantique > 12 (vs 8.0 avec MiniLM)

### B.2. Décodeur BGE-M3
- **Ce qu'on fait** : Ré-entraîner le décodeur GPT-2 avec les embeddings BGE-M3.
- **Commande** : `python -u -m msl.train.train_text_decoder --corpus runs/bge_corpus.pt --steps 20000 --lr 3e-5`
- **Statut** : Code prêt (auto-détecte 1024-dim)
- **Critère** : 5/5 paraphrases fidèles

### B.3. LLM natif BGE-M3
- **Ce qu'on fait** : Entraîner le LLM natif sur les paquets BGE-M3.
- **Statut** : À lancer après B.2
- **Critère** : > 85% codes corrects

### B.4. Comparaison MiniLM vs BGE-M3
- **Ce qu'on fait** : Comparer les deux pipelines sur tous les critères.
- **Métriques** : Qualité paraphrases, gap sémantique, gain vitesse, mémoire
- **Décision** : Choisir le meilleur encodeur pour la suite.

---

## Phase C — Canal littéral (preservation des valeurs exactes)

### C.1. Extracteur de littéraux
- **Ce qu'on fait** : Détecter et extraire les valeurs exactes (noms, nombres, dates, identifiants) du texte.
- **Méthode** : NER (Named Entity Recognition) + regex pour nombres/dates.
- **Sortie** : Table de littéraux référencée par position dans les paquets.
- **Statut** : Pas commencé
- **Critère** : > 95% des valeurs exactes correctement extraites

### C.2. Intégration au codec
- **Ce qu'on fait** : Le codec produit deux choses : paquets MSL (sens) + table de littéraux (valeurs).
- **Format** :
  ```
  msl_message = {
      "packets": [(...), (...)],           # le sens (MSL discret)
      "literals": ["Paris", "2026-07-30", 616],  # les valeurs exactes
      "references": [...]                  # alignement paquets <-> littéraux
  }
  ```
- **Statut** : Pas commencé
- **Critère** : Round-trip 100% sur les valeurs exactes

### C.3. Décodeur avec canal littéral
- **Ce qu'on fait** : Le décodeur GPT-2 reçoit les paquets MSL **et** la table de littéraux.
- **Conditionnement** : Prefix embeddings (paquets) + literal embeddings (table).
- **Statut** : Pas commencé
- **Critère** : "On July 18th, Muiriel turned 25" → round-trip exact (pas de paraphrase sur les valeurs)

### C.4. LLM natif avec canal littéral
- **Ce qu'on fait** : Le LLM natif prédit les paquets MSL **et** référence les littéraux.
- **Statut** : Pas commencé
- **Critère** : Génère du contenu qui préserve les valeurs exactes

---

## Phase D — Passage a l'echelle (apres validation)

### D.1. Plus de données
- 1M phrases (Tatoeba + Wikipedia + OpenWebText)
- Multilingue (FR + EN + ES + DE)

### D.2. Plus gros modèle
- GPT-2 medium (350M) ou large (770M)
- Ou LLaMA-3 1B/3B si disponible en open weights

### D.3. Plus de steps
- 100k+ steps
- Curriculum learning (phrases simples → complexes)

### D.4. Évaluation humaine
- BLEU, BERTScore, évaluation humaine
- Test de traduction FR ↔ EN via MSL

---

## Résumé du plan

| Phase | Ce qu'on fait | Quand | Durée estimée |
|---|---|---|---|
| **A** | Pipeline complet MiniLM | Maintenant | ~2h sur A10 |
| **B** | BGE-M3 + comparaison | Après A | ~3h sur A10 |
| **C** | Canal littéral | Après choix encodeur | ~1 jour de dev + A10 |
| **D** | Passage à l'échelle | Après C validé | Jours → semaines |

## Règle de décision

1. **On ne passe à la phase suivante que si la précédente est validée.**
2. **On mesure à chaque étape** (pas de supposition).
3. **On documente tout** (chaque résultat va dans `docs/`).
4. **Le canal littéral n'est ajouté qu'après le choix définitif de l'encodeur** (MiniLM ou BGE-M3).

## Ce qui tourne maintenant

```
A.1 — Décodeur MiniLM sur A10
       20k steps, ~36 min
       loss actuelle : 0.51 (step 10000)
```

Ensuite : A.2 → A.3 → A.4 → A.5 → B.1 → B.2 → B.3 → B.4 → C.
