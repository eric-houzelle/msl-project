# H2 : MSL est un standard adoptable

Auteur : Kimi K3 (rôle)
Date : 31 juillet 2026
Objet : Test de l'hypothèse H2 (standard inter-modèles). Verdict : MSL est adoptable comme standard.

---

## 1. La question

MSL est-il un langage partagé ou un code privé ? Deux tests :

- **Émergence** : deux codecs indépendants convergent-ils vers le même langage ?
- **Adoption** : un nouveau modèle peut-il apprendre un langage figé ?

## 2. Test 1 : Émergence

Deux codecs PQ (48 codebooks × 256, seeds 0 et 42) entraînés indépendamment sur 2000 phrases Tatoeba.

### Résultats

| Métrique | Valeur | Chance | Ratio |
|---|---|---|---|
| Même voisin sémantique | 21,1 % | 0,05 % | **422×** |
| Voisin dans le top-10 | 37,1 % | 0,5 % | **74×** |
| Corrélation des matrices de distance | 0,281 | 0,0 | positive |
| Match des codes | 0,3 % | 0,4 % | nul |

### Interprétation

Les deux codecs **ne produisent pas les mêmes codes** (0,3 % de match, soit le hasard). C'est attendu : des initialisations différentes donnent des codebooks différents.

Mais ils **trouvent les mêmes voisins sémantiques** dans 21 % des cas (422× la chance). Exemple : "I have to go to sleep" → les deux codecs trouvent indépendamment "I have to go to bed" comme voisin le plus proche.

**Le langage émerge partiellement.** La structure sémantique est partiellement partagée, même si les codes exacts diffèrent. C'est comme deux personnes qui, sans s'être jamais rencontrées, classent les mêmes objets dans les mêmes catégories mais utilisent des mots différents pour les nommer.

### Corrélation 0,28
La corrélation des matrices de distance est 0,28 — positive mais faible. Les codecs sont d'accord sur la structure grossière mais pas sur les détails. Pour un standard fort, il faudrait > 0,5.

## 3. Test 2 : Adoption

On fige un codec comme "MSL v0" (le standard). On entraîne un **nouvel encodeur** (init aléatoire, 2,6M params) à produire des embeddings que le codec figé quantize correctement. 3000 steps.

### Résultats

| Métrique | Valeur | Chance | Ratio |
|---|---|---|---|
| Agreement des paquets | **86,9 %** | 0,4 % | **217×** |
| Même voisin que le standard | **45,1 %** | 0,05 % | **900×** |

### Interprétation

Un nouvel encodeur, parti de zéro, apprend à produire des paquets **compatibles avec le standard figé** à 86,9 % de match. Et il retrouve les mêmes voisins sémantiques dans 45 % des cas.

Exemples :
- "Let's try something" → standard et nouvel encodeur trouvent tous les deux "It's not something anyone can do"
- "I have to go to sleep" → les deux trouvent "I have to go to bed"

**Le standard est adoptable.** Un nouveau modèle peut apprendre MSL v0, comme on apprend une langue étrangère. Il ne l'invente pas — il l'apprend.

## 4. Verdict H2

| Test | Résultat | Verdict |
|---|---|---|
| Émergence | 21 % convergence, corrélation 0,28 | **Partielle** — le langage émerge mais n'est pas parfait |
| Adoption | 87 % agreement, 45 % même NN | **Forte** — le standard est apprenable |

### Conclusion

**MSL est un standard adoptable.**

1. L'émergence naturelle est **partielle** (21 %) — insuffisante pour un standard spontané, mais le signal est 422× la chance. La structure sémantique est partiellement partagée.

2. L'adoption est **forte** (87 %) — un nouveau modèle peut apprendre le standard figé. C'est la voie recommandée par le brief (§7.2) : figer le codec, enseigner aux nouveaux modèles.

3. La stratégie est claire : **standardiser, pas attendre l'émergence**. On fige MSL v0, on l'enseigne. C'est comme TCP/IP — on ne demande pas à tout le monde de réinventer le protocole, on fixe le standard et tout le monde l'apprend.

## 5. Comparaison avec le brief

Le brief (§7.2) dit : « New emitter or receiver → frozen MSL codec → conformity tests ». C'est exactement le test 2. Résultat : 87 % de conformité. Le standard est adoptable.

Le brief (§12, risque "Instabilité du langage") dit : « Chaque entraînement invente un protocole incompatible ». C'est partiellement confirmé par le test 1 (0,3 % de match des codes), mais contredit par la convergence des voisins (21 %). La parade du brief (codec figé + conformité) est validée par le test 2 (87 %).

## 6. Ce que ça change pour le projet

**La pièce manquante est en place.** On a maintenant :
- H1 (paquets portent le sens) : validée
- H2 (standard adoptable) : **validée**
- H3 (gain économique) : validée (6× plus rapide)

Le triangle est complet. Le concept MSL tient sur ses trois piliers.

## 7. Artefacts

- `src/msl/eval/test_h2.py` : les deux tests H2.
- `runs/h2_test.log` : résultats complets.
