# Passage au vrai texte — MSL capture le sens

Auteur : Kimi K3 (rôle)
Date : 31 juillet 2026
Objet : Premier test de MSL sur du vrai texte humain (Tatoeba, 10 000 phrases en anglais). MSL capture le sens, mais la résolution est encore grossière.

---

## 1. Ce qu'on a fait

### L'encodeur
On utilise `all-MiniLM-L6-v2` (22M params, pré-entraîné sur des millions de phrases). Il transforme une phrase en un embedding de 384 dimensions qui capture le sens : "Paris is the capital of France" et "France capital is Paris" ont une similarité cosinus de 0.986.

### Le dataset
10 000 phrases en anglais simple de Tatoeba (vraies phrases, vraie diversité) :
- "Let's try something."
- "I have to go to sleep."
- "The password is Muiriel."
- etc.

### Le quantizer sémantique
On apprend un PQ quantizer (48 codebooks × 256 entrées = 384 bits) avec une **perte de préservation sémantique** : les phrases sémantiquement similaires doivent avoir des paquets similaires.

## 2. Les résultats

### Résolution sémantique

| Config | Bits | Distance similaire | Distance aléatoire | Gap |
|---|---|---|---|---|
| PQ seul (16×256) | 128 | 11.7/16 | 15.8/16 | 4.1 |
| PQ seul (48×256) | 384 | 43.5/48 | 47.8/48 | 4.3 |
| **PQ + sémantique (48×256)** | **384** | **39.6/48** | **47.6/48** | **8.0** |

La perte sémantique double le gap : de 4.3 à 8.0 codes de différence. Les phrases similaires sont maintenant clairement plus proches en MSL.

### Récupération (nearest neighbor)

15% des phrases retrouvent leur voisin sémantique dans le top-10. Un exemple parfait :

| Requête | Voisin embedding | Voisin paquets MSL |
|---|---|---|
| "I have to go to sleep." | "I have to go to bed." | **"I have to go to bed."** |

Le système a trouvé **exactement** le bon voisin sémantique via les paquets. Mais sur 20 phrases, seules 3 ont réussi. La résolution est encore insuffisante pour une récupération fiable.

## 3. Interprétation

### Ce qui marche
- **MSL capture le sens du vrai texte** : les phrases similaires ont des paquets plus proches que les phrases aléatoires.
- **La perte sémantique aide** : le gap double (4→8 codes).
- **La récupération marche parfois** : le système retrouve le bon voisin sémantique dans 15% des cas.

### Ce qui ne marche pas encore
- **Résolution insuffisante** : 8 codes sur 48 (17%) n'est pas assez pour distinguer finement le sens. Il faudrait un gap de 20+ pour une récupération fiable.
- **15% de récupération** : trop faible pour être utile. Le système confond des phrases très différentes.
- **Pas de décodeur texte** : on ne génère pas encore du texte depuis les paquets, on fait juste du nearest-neighbor retrieval.

## 4. Ce qu'il faudrait pour améliorer

1. **Plus d'entraînement** : 5000 steps n'est pas assez. 50k+ steps avec un lr schedule adaptatif.
2. **Quantizer adaptatif** : allouer plus de codes aux phrases difficiles à distinguer (au lieu de PQ uniforme).
3. **Plus de codebooks** : 96 au lieu de 48 (768 bits) donnerait plus de résolution.
4. **Joint training** : entraîner l'encodeur et le quantizer ensemble (au lieu de figer l'encodeur).
5. **Décodeur** : un petit LM qui génère du texte depuis les paquets (au lieu du nearest-neighbor).

## 5. Ce que ça prouve

**MSL fonctionne sur du vrai texte.** Ce n'est pas un artefact du monde synthétique. Le sens est capturé dans les paquets, même si la résolution est grossière. C'est le premier pas hors du synthétique.

## 6. Artefacts

- `runs/realtext_corpus.pt` : 10 000 phrases + embeddings + paquets (PQ seul).
- `runs/realtext_semantic.pt` : même chose avec quantizer sémantique (gap 8.0).
- `src/msl/train/train_semantic_quantizer.py` : quantizer sémantique.

## 7. Reproductibilité

```bash
# Construire le corpus
(.venv) python -u -m msl.data.build_realtext_corpus

# Entraîner le quantizer sémantique
(.venv) python -u -m msl.train.train_semantic_quantizer
```
