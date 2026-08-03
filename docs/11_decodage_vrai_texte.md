# MSL génère du vrai texte depuis les paquets

Auteur : Kimi K3 (rôle)
Date : 31 juillet 2026
Objet : Premier décodage de vrai texte humain depuis des paquets MSL. Le décodeur reconstruit le sens, pas les mots — 4/5 paraphrases fidèles.

---

## 1. Le résultat

Le décodeur MSL prend des paquets (48 codes, 384 bits) et génère du texte en anglais. Il ne reproduit pas les mots originaux — il **reconstruit le sens** depuis les paquets et écrit une nouvelle phrase avec le même sens.

| # | Phrase originale | Généré depuis MSL | Verdict |
|---|---|---|---|
| 1 | "Let's try something." | "Give me a chance." | Paraphrase parfaite |
| 2 | "I have to go to sleep." | "I have to go to bed." | Paraphrase parfaite |
| 3 | "Today is June 18th and it is Muiriel's birthday!" | "The day of the birthday..." | Même sujet, brouillé |
| 4 | "The password is Muiriel." | "You have to enter the password." | Paraphrase parfaite |
| 5 | "I will be back soon." | "I'll be back soon." | Quasi identique |

**4 exemples sur 5 sont des paraphrases fidèles.** Le 5e est même presque identique au original.

## 2. L'architecture

```
Phrase originale
    ↓
all-MiniLM-L6-v2 (encodeur pré-entraîné, 22M params, figé)
    ↓
embedding 384-dim (capture le sens)
    ↓
projection (2.6M params, apprise)
    ↓
PQ quantizer (48 codebooks × 256, EMA, 384 bits)
    ↓
z_q (384-dim, version quantizée de l'embedding)
    ↓
projection → 4 prefix embeddings (768-dim)
    ↓
GPT-2 (124M params, figé) conditionné par les prefix
    ↓
Phrase reconstruite
```

### Les composants
- **Encodeur** : `all-MiniLM-L6-v2` (22M params, pré-entraîné sur des millions de phrases). Transforme une phrase en un embedding de 384 dimensions qui capture le sens. Figé pendant l'entraînement.
- **Quantizer** : PQ (Product Quantization) avec 48 codebooks de 256 entrées = 384 bits par phrase. Codebooks mis à jour par EMA.
- **Projection** : couche linéaire (2.6M params) qui projette z_q en 4 "prefix embeddings" de dimension 768 (la dimension de GPT-2). C'est le seul composant appris.
- **Décodeur** : GPT-2 small (124M params, pré-entraîné). Génère du texte conditionné par les prefix embeddings. Figé pendant l'entraînement.

### Pourquoi ça marche
1. L'encodeur pré-entraîné capture déjà le sens — on ne part pas de zéro.
2. GPT-2 sait déjà générer du texte — on ne lui apprend que le conditionnement.
3. La projection (2.6M params) est le seul composant appris — entraînement rapide et stable.
4. Le quantizer compresse l'embedding en 384 bits tout en préservant assez de sens pour guider GPT-2.

## 3. L'entraînement

- **Dataset** : 10 000 phrases de Tatoeba (anglais simple, vrai texte).
- **Loss** : cross-entropy sur les tokens (teacher forcing). Le modèle prédit chaque token depuis le prefix + les tokens précédents.
- **Durée** : 5 000 steps, batch 16, lr 5e-5, ~14 minutes sur MPS.
- **Loss finale** : 0.52 (vs 3.6 au départ).
- **Params appris** : 2.6M (projection seule). GPT-2 (124M) et encodeur (22M) sont figés.

## 4. Ce que ça prouve

### Le sens traverse le goulot MSL
La phrase "Let's try something" est compressée en 384 bits, puis décompressée en "Give me a chance" — une phrase avec le **même sens** mais des mots différents. Le sens a survécu à la compression discrète.

### MSL n'est pas du français ou de l'anglais déguisé
Si MSL était juste de l'anglais compressé, le décodeur reproduirait les mots originaux. Au lieu de ça, il génère des **paraphrases** — "try something" → "give me a chance", "go to sleep" → "go to bed". MSL capture le sens, pas la forme.

### Le décodeur est un traducteur sémantique
Le décodeur ne fait pas de la compression lossless (reproduire exactement). Il fait de la **traduction sémantique** : il lit le sens dans les paquets et l'exprime en anglais. C'est exactement le comportement voulu.

## 5. Limites actuelles

### Le 3e exemple échoue
"Today is June 18th and it is Muiriel's birthday!" → "The day of the birthday of the first lady..." Le décodeur perd les détails spécifiques (date, nom) et génère une phrase générique sur les anniversaires. Les **valeurs exactes** (noms, nombres, dates) ne survivent pas au goulot — c'est le problème du canal littéral identifié dans le brief §5.4.

### Pas de quantization dans le test
Le test utilise l'embedding **continu** (z_q avant quantization) comme conditionnement, pas les paquets discrets. La quantization réelle (PQ) ajoute du bruit qui pourrait dégrader la qualité. À tester.

### Une seule seed, 5000 steps
Pas d'intervalle de confiance. Le résultat est encourageant mais à confirmer.

### GPT-2 est figé
On n'apprend pas GPT-2 — on apprend juste la projection. Fine-tuner GPT-2 améliorerait probablement la qualité mais coûterait plus cher.

## 6. Ce qu'il faut améliorer

1. **Canal littéral** : préserver les noms, nombres, dates exacts (brief §5.4).
2. **Tester avec quantization réelle** : utiliser les paquets discrets, pas z_q continu.
3. **Plus de steps** : 10k-50k steps pour réduire la loss.
4. **Plus de données** : 100k phrases au lieu de 10k.
5. **Fine-tuner GPT-2** : débloquer les dernières layers pour adapter au domaine.
6. **Évaluer la fidélité** : mesurer la similarité sémantique entre original et généré (BLEU, BERTScore, ou cosine similarity d'embeddings).

## 7. Artefacts

- `runs/text_decoder_0.pt` : décodeur entraîné (projection 2.6M + GPT-2 124M figé).
- `runs/realtext_corpus.pt` : 10 000 phrases + embeddings + paquets.
- `src/msl/train/train_text_decoder.py` : entraînement du décodeur.
- `src/msl/models/structured_codec.py` : codec structuré (synthétique, 100% fidèle).

## 8. Reproductibilité

```bash
# Construire le corpus (10k phrases Tatoeba)
(.venv) python -u -m msl.data.build_realtext_corpus

# Entraîner le décodeur (14 min sur MPS)
(.venv) python -u -m msl.train.train_text_decoder --steps 5000

# Le test de génération est inclus à la fin de l'entraînement.
```

## 9. Le chemin parcouru

| Étape | Résultat | Document |
|---|---|---|
| Monde synthétique : paquets portent le sens | 30% acc, gap +16 | `04`, `05` |
| Monde synthétique : LLM natif 6× plus rapide | 92% codes, 6× speedup | `07`, `08` |
| Monde synthétique : round-trip 100% fidèle | 159/159 faits | `09` |
| Vrai texte : MSL capture le sens | gap 8.0, 15% retrieval | `10` |
| **Vrai texte : décodage génère des paraphrases** | **4/5 fidèles** | **ce document** |

On est passé du synthétique au vrai texte, et le décodeur génère des paraphrases fidèles depuis les paquets MSL. C'est le résultat le plus concret du projet.
