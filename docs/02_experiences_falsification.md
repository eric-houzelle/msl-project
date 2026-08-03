# Cinq expériences de falsification — MSL

Auteur : Kimi K3 (rôle)
Date : 31 juillet 2026
Objet : Spécifier une expérience **minimale, exécutable et falsifiable** pour chacune des cinq hypothèses identifiées dans `01_audit_critique.md`. Chaque expérience est conçue pour **tuer** son hypothèse si elle échoue, pas pour prouver qu'elle marche. Conformément au brief §15, on ne code pas encore le système complet — on spécifie le protocole.

Convention : une hypothèse est **réfutée** si on peut rejeter l'hypothèse nulle H₀ ci-dessous au seuil α = 0.05 (test bilatéral, sauf indication) avec une puissance ≥ 0.8. Toutes les expériences partagent le même monde synthétique (cf. §0) pour permettre la réutilisation des seeds et des splits.

---

## 0. Monde synthétique commun (MS-1)

Toutes les expériences sauf la H4 s'appuient sur **MS-1**, un générateur déterministe et infini d'« états du monde ».

### Structure d'un état

Un état *s* ∈ MS-1 est un graphe attribué :

```
s = (entités, relations, attributs, événements, modalités, littéraux)

entités    = {e₁, …, e_n}       ; n ∈ [1, 8]
relations  = {(e_i, type, e_j)}  ; type ∈ {cause, avant, contient, …}  (12 types)
attributs  = {(e_i, k, v)}      ; k ∈ 20 clés, v ∈ domaine discret
événements = {(t, e_i, action)}  ; t ∈ horodatage, action ∈ 15 actions
modalités  = {réalisé, envisagé, nié, incertain, rapporté}  — combinatoires
littéraux  = {noms, nombres, dates} marqués exacts
```

### Complexité paramétrable

`difficulté(s) = k` = nombre d'attributs+relations+événements (k ∈ {2, 4, 8, 16, 32, 64}).

Le générateur produit des états *inédits* sur demande (tirage pseudo-aléatoire avec graine), ce qui élimine le risque de mémorisation et permet des splits train/test propres.

### Tâches

Chaque état s supporte un ensemble de **questions** à vérité de terrain exacte, évaluables par exécution :

- QA factuel : « que vaut l'attribut A de e_i ? »
- Implication : « est-ce que e_i cause e_j ? »
- Contradiction : « est-ce que s est compatible avec s' ? »
- Temps : « e_i est-il avant e_j ? »
- Composition : « si on ajoute l'événement x à s, que devient l'attribut A ? »
- Valeurs exactes : récupération d'un littéral indexé.

**Clef** : la vérité de terrain est **calculée**, pas annotée — on peut générer infiniment et sans biais d'annotation.

### Taille initiale du vocabulaire d'attributs

20 clés × ~8 valeurs moyennes ≈ 160 atomes sémantiques, plus 12 types de relations, 15 actions, 5 modalités. C'est volontairement petit pour que H1 et H6 soient *testables* sans confondre l'échec avec un manque de données.

---

## H1 — Un paquet discret opaque peut porter un état complet, pas un concept isolé

### H₀ (nulle)
La longueur MSL minimale *L(s)* pour reconstruire parfaitement l'état *s* (toutes tâches résolues) **croît linéairement** avec le nombre d'atomes sémantiques *k* : `L(s) = a·k + b`, avec `a ≥ a_text` (a_text = longueur textuelle par atome).

### H₁ (alternative)
`L(s)` croît **sous-linéairement** (logarithmique ou plateau) : un paquet code plusieurs atomes de façon distribuée.

### Dispositif

1. Encoder-décoder (E, Q, D) entraînés sur MS-1 à reconstruire l'état, sans canal littéral (afin d'isoler la capacité du paquet).
2. Geler le codec.
3. Pour k ∈ {2, 4, 8, 16, 32, 64}, mesurer *L(s)* minimal garantissant récupération parfaite (tâches à 100 %) sur 1000 états inédits par palier.

### Métrique primaire
Pente `a` de la régression `L(s) ~ a·k + b`. Rapport `a / a_text`.

### Critère de falsification (go/no-go)
- **H1 réfutée** si `a / a_text ≥ 0.8` sur la plage k ∈ [4, 32] (intervalle de confiance 95 %). Interprétation : le paquet ne compresse rien, c'est un token-par-atome déguisé.
- **H1 confortée** si `a / a_text ≤ 0.4` sur la même plage.
- Zone grise (0.4 < a/a_text < 0.8) : investiguer avant de conclure (vérifier le collapse de codebook, ajuster B et V).

### Ablations requises
- Varier B ∈ {4, 8, 16} codebooks × V ∈ {256, 1024, 4096} pour séparer l'effet « paquet plus large » de l'effet « distribution vraie ».
- Comparer PQ vs RVQ vs FSQ (brief §5.2) — si tous échouent pareil, la cause est structurelle, pas un mauvais choix de quantifier.
- Baseline *triviale* : un encodeur qui émet un code par atome (one-hot) doit être *battue* par MSL — sinon H1 est trivialement fausse.

### Risque de faux négatif
Si le collapse de codebook réduit la capacité effective, on peut conclure à tort que H1 est fausse. **Parade** : mesurer l'utilisation des codebooks (perplexité des codes, codebook usage) ; exiger ≥ 80 % de codes actifs avant d'accepter le verdict.

### Budget
≈ 3 runs × 6 paliers × 3 quantifiers = ~54 entraînements d'un petit encodeur (≤ 30 M params). Sur 1 GPU unique, ~12 h par run → 1 semaine.

---

## H2 — Le langage émergent est non privé, généralisable à de nouveaux participants

### H₀
Un récepteur entraîné sur un codec A et testé sur un codec B (indépendant, même architecture, même MS-1) **n'obtient pas de résultat meilleur que le hasard** sur les tâches.

### H₁
Le récepteur transfère : son taux de réussite dépasse le hasard d'une marge significative.

### Dispositif

1. Entraîner **N = 8 émetteurs** indépendants (seeds différentes, init différentes, ordres de données différents) sur MS-1, sans canal littéral.
2. Pour chaque paire (A, B) avec A ≠ B : geler le codec A, entraîner un récepteur **naïf** (init aléatoire) qui ne voit **que les paquets A** et doit reconstruire l'état — c'est la baseline intra-codec.
3. **Transfert inter-codec** : geler B, entraîner un récepteur déjà entraîné sur A, tester sur B **sans réentraînement** sur B (zero-shot), puis avec **fine-tuning léger** sur B (5 % du budget initial).
4. Répéter avec un **récepteur fraîchement initialisé** sur B (baseline de chance).
5. Mesurer le taux de réussite aux tâches (QA, implication, contradiction).

### Métrique primaire
`T_inter = réussite moyenne (zero-shot inter-codec)`
`T_intra = réussite moyenne (intra-codec)`
`T_chance = réussite moyenne (récepteur fraîchement initialisé, intra-codec, avant entraînement)`

Indices secondaires :
- **PosDis** (positional disentanglement, Chaabouni et al.) pour mesurer la compositionnalité positionnelle.
- **BAscore** (Bottleneck Alignment, auto-alignment de codebooks) pour quantifier la similarité entre A et B.

### Critère de falsification
- **H2 réfutée** si `T_inter ≤ T_chance + ε` avec ε = 5 points, en zero-shot et après fine-tuning léger. Interprétation : les codecs sont privés, il n'y a pas de standard.
- **H2 confortée** si `T_inter ≥ 0.7 · T_intra` en zero-shot, ou si `T_inter ≥ T_intra − 2 points` après fine-tuning léger.
- Critère dur : `T_inter ≥ T_intra − 2 pts` **sans** fine-tuning → standard *fort* ; avec fine-tuning léger → standard *faiblement alignable* (encore utile).

### Ablation
- Multi-agent vs mono-agent : inclure une condition où A et B sont entraînés *ensemble* (jeu de référence, §7.1 du brief). Si seul le multi-agent donne un standard, c'est conforme à Kaszyński 2026.
- Iterated learning : périodiquement remplacer l'émetteur le plus vieux par un nouveau, pour imposer la reproductibilité inter-générations (cf. Kirby,_cleanup).

### Risque de faux positif
Si A et B convergent vers le *même* code par hasard (peu probable avec init indépendantes, mais possible sur MS-1 si le domaine est trop petit), on peut conclure à tort que le standard existe. **Parade** : MS-1 est volontairement ouvert (génération infinie), et on mesure la diversité des codes actifs — si elle est trop faible, le test est non informatif.

### Budget
8 émetteurs × 1 récepteur each + ~8×7 = 56 paires × 1 test ≈ 100 entraînements petits. ~2 semaines sur 1 GPU.

---

## H3 — La réduction de tokens se traduit en gain réel de FLOPS/mémoire, E/Q/D inclus

### H₀
À FLOPS d'entraînement **et** d'inférence égalisés, le système MSL ne dépasse pas la baseline texte sur les tâches MS-1 (Δ qualité ≤ 0).

### H₁
À FLOPS égal, MSL obtient une qualité supérieure, ou obtient la même qualité avec strictement moins de FLOPS.

### Dispositif

1. Définir **3 baselines** :
   - **B-text** : Transformer texte tokenisé (BPE, V=8k), entraîné sur la formulation textuelle de MS-1.
   - **B-latent-cont** : Transformer sur latent **continu** (type Coconut), sans goulot discret.
   - **B-latent-discret-RVQ-rigid** : MSL avec quantifier rigide (un code par atome, pas de distribution apprise) — borne inférieure.
2. Entraîner MSL et les baselines à **taille de paramètres comparable** (~150 M).
3. Mesurer, pour chaque système :
   - FLOPS d'entraînement (via `fvcore` ou `thop`).
   - FLOPS d'inférence par tâche.
   - Pic mémoire GPU (entraînement et inférence).
   - KV-cache à longueur informationnelle égale.
   - Latence par tâche (ms).
   - Énergie (estimée via `carbontracker` ou mesure GPU si disponible).
4. **Égalisation** : pour chaque système, ajuster le budget d'entraînement pour que les FLOPS totaux tombent dans une fenêtre ±10 % d'une cible commune. Pour l'inférence, comparer à qualité égale (courbe débit-distorsion).

### Métrique primaire
`Δ = qualité(MSL, FLOPS=cible) − qualité(B-text, FLOPS=cible)`
à la **même** cible FLOPS, sur le même ensemble de tâches.

Métriques secondaires :
- Pareto frontière (qualité, FLOPS inférence).
- Ratio KV-cache MSL / B-text à longueur informationnelle égale.

### Critère de falsification
- **H3 réfutée** si `Δ ≤ 0` pour au moins **2 cibles FLOPS distinctes** (par exemple 1e17 et 1e18 FLOPS entraînement). Interprétation : le gain de tokens ne se traduit pas en gain économique.
- **H3 confortée** si `Δ > 0` sur **au moins une** cible, ET la courbe débit-distorsion MSL domine celle de B-text (même qualité à ≥ 25 % moins de bits de sortie).

### Ablation
- Isoler le coût de E/Q/D : mesurer FLOPS de la seule étape T (prédiction MSL), puis FLOPS end-to-end. Le gain doit persister end-to-end, pas seulement au cœur.
- Varier la longueur de séquence MSL (court/moyen/long) pour vérifier que le gain n'est pas un artefact de séquences courtes.

### Risque de faux négatif
Si E/Q/D sont mal optimisés, le bilan end-to-end peut être négatif même si T est gagnant. **Parade** : reporter deux bilans — « cœur seul » et « end-to-end » — et conclure H3 sur le end-to-end mais documenter le cœur.

### Risque de faux positif
Si la baseline texte est sous-entraînée, MSL peut *sembler* gagner. **Parade** : la baseline texte doit être entraînée jusqu'à convergence (loss plateau) avant toute comparaison.

### Budget
4 systèmes × 2 cibles FLOPS × 3 seeds = 24 entraînements à 150 M. ~4 semaines sur 1 GPU.

---

## H4 — Un Transformer sans texte reste compétitif sur des tâches sémantiques

### H₀
À taille égale, un Transformer T_MSL entraîné *exclusivement* sur des paquets MSL (jamais vu de texte) **n'égale pas** un Transformer T_text de même taille entraîné sur le texte correspondant, sur des tâches dérivées du même contenu.

### H₁
T_MSL égale ou dépasse T_text, ou atteint la même qualité avec strictement moins de FLOPS/mémoire (ceci recoupe partiellement H3, mais H4 porte sur la *capacité*, pas l'économie).

### Dispositif

1. Phase 2 (figer codec MSL sur MS-1).
2. **Deux domaines** :
   - **MS-1 seul** (domaine synthétique, contrôlé).
   - **MS-1 + petits textes multilingues** (extension contrôlée vers le langage, cf. §8.1 ordre 4).
3. Entraîner :
   - T_MSL : Transformer ~150 M, corpus = paquets MSL uniquement.
   - T_text : Transformer ~150 M, corpus = formulation textuelle de **même contenu**.
   - T_mix : (baseline anti-réfutation, **hors contrainte principale**) Transformer mixant texte et MSL — **pour mesurer le coût de la contrainte « sans texte »**, pas pour la valider.
4. Évaluer sur :
   - Tâches MS-1 (QA, implication, contradiction, temps, composition, littéraux).
   - Pour le domaine étendu : paraphrases, multilingue, calcul.

### Métrique primaire
`Gap = qualité(T_text) − qualité(T_MSL)` sur l'ensemble des tâches.

### Critère de falsification
- **H4 réfutée** si `Gap ≥ 3 points` sur MS-1 seul (zone où la parité est *attendue*). Interprétation : même sur un domaine contrôlé, priver le Transformer de texte coûte trop.
- **H4 confortée sur MS-1** si `Gap < 1 point`.
- **H4 confortée sur langage** si `Gap < 5 points` sur le domaine étendu (on accepte plus de perte sur le langage ouvert).
- **H4mise en échec partielle** si MS-1 est ok mais domaine étendu > 5 pts : documenter comme « plafond de concept » et continuer sans prétendre à la parité sur langage ouvert avant nouvelle preuve.

### Ablation
- Varier la taille du codebook (B et V) : si augmenter B réduit le Gap, c'est un problème de capacité, pas de principe.
- Comparer T_MSL au T_latent-continu (Coconut-like) : si T_MSL < T_latent-continu, le goulot discret coûte trop — distinct de H4 mais corrélé.

### Risque de faux négatif
Si le codec MSL figé est de mauvaise qualité, T_MSL échoue par la faute de Q, pas par la faute de la contrainte. **Parade** : exiger que le codec MSL ait passé H1 avant d'entraîner T_MSL.

### Budget
3 modèles × 2 domaines × 3 seeds = 18 entraînements à 150 M. ~6 semaines sur 1 GPU.

---

## H5 — Le canal littéral peut être borné sans devenir trou

### H₀
Quand on restreint le budget du canal littéral, le système **compense** en augmentant le volume de littéraux jusqu'à saturer le budget, plutôt qu'en compressant le latent — i.e. le canal *détourne* la compression.

### H₁
L'usage du canal croît avec le **nombre de valeurs exactes** (et non avec le volume de contenu global), et reste sous le quota.

### Dispositif

1. Construire une variante de MS-1 où l'on contrôle indépendamment :
   - `C` = volume de contenu (atomes sémantiques non-exacts).
   - `L` = nombre de valeurs exactes marquées (noms, nombres, dates).
2. Entraîner le système MSL complet (E, Q, D, canal littéral) avec différents quotas `Q_lit` sur le canal (0, 25 %, 50 %, 100 % du budget total).
3. Mesurer :
   - `V_lit(C, L)` = volume moyen du canal.
   - `Err_exact` = taux d'erreur sur les littéraux.
   - `Compression_ratio` = bits totaux / entropie de la source.

### Métrique primaire
Régression `V_lit = α·L + β·C + γ`. On attend `α > 0` et `β ≈ 0`.

### Critère de falsification
- **H5 réfutée** si `β > α/2` (le canal croît avec le contenu général, pas avec les valeurs exactes). Interprétation : le canal est un trou, la compression est contournée.
- **H5 confortée** si `β ≈ 0` et `Err_exact < 1 %` sous le quota nominal.
- **H5mise en échec par sur-strict** : si `Err_exact > 10 %` sous quota nul, le système *préfère mal deviner que d'utiliser le canal* — il faut ajuster la pénalité.

### Ablation
- Varier la pénalité du canal dans `L_total` (brief §6.2, `delta`).
- Comparer « canal libre » vs « canal avec quotas durs » vs « canal avec pénalité seule » — distinguer l'effet incitation de l'effet contrainte.

### Risque de faux positif
Le système peut *apprendre à copier les littéraux* sans les marquer comme tels (les injecter dans le latent). **Parade** : sonder le latent avec un classificateur externe pour vérifier qu'aucun littéral n'est récupérable — s'il l'est, le canal est contourné dans l'autre sens.

### Budget
4 quotas × 4 configurations (C, L) × 3 seeds = ~48 entraînements petits. ~1 semaine.

---

## Récapitulatif et ordre d'exécution

| Hypothèse | Risque | Effet de l'échec | Budget GPU | À lancer |
|---|---|---|---|---|
| H1 (état complet) | Haut | Tue le projet structurellement | ~1 sem | **Phase 1, dès le départ** |
| H2 (standard non privé) | Haut | Tue le projet structurellement | ~2 sem | **Phase 1, parallèle à H1** |
| H3 (gain de FLOPS) | Haut | Tue la rentabilité, pas la recherche | ~4 sem | **Phase 1 (prélim) → Phase 2** |
| H4 (sans texte) | Moyen/haut | Limite la portée, pas l'existence | ~6 sem | **Phase 3**, après Phase 2 |
| H5 (canal littéral) | Moyen | Limitation opérationnelle | ~1 sem | **Phase 2**, parallèle à H3 |

**Séquence recommandée** :
1. H1 + H2 en parallèle (Phase 1). Si l'un des deux est réfuté, **stop**.
2. H5 en parallèle de Phase 2.
3. H3 en Phase 1-préliminaire puis Phase 2 (à figer).
4. H4 seulement si Phase 2 a produit un codec figé convaincant.

**Critère go/no-go global après Phase 1** :
- Si H1 **et** H2 sont confortées (zone verte) → **go** Phase 2.
- Si l'une des deux est en zone grise → investiguer, ne pas avancer.
- Si l'une des deux est réfutée → **no-go** ou reformuler le projet (cf. §7 du brief, « proposer plusieurs variantes lorsqu'un choix n'est pas encore tranché »).
