# Audit critique — Machine Semantic Language (MSL)

Auteur : Kimi K3 (rôle)
Destinataire : Eric Houzelle
Date : 31 juillet 2026
Objet : Audit scientifique du brief `Brief_projet_MSL_Kimi_K3.docx`, conformément à la section 15 (« Commence par auditer ce brief »).

---

## 1. Verdict de faisabilité

Le projet est **directionnellement sain et défendable**, mais sa formulation actuelle empaquette **trois programmes de recherche distincts** en un seul :

1. un **codec sémantique discret** (encodeur E + goulot Q + décodeur D) ;
2. un **protocole de communication émergent multi-agents** (émetteur/récepteur/vérificateur) ;
3. un **Transformer natif latent** (cœur n'ayant jamais vu de texte).

Aucun des trois n'est hors de portée individuellement — chacun a des précurseurs publiés qui valident des briques. Le risque vient de leur **jonction simultanée** et de l'exigence « aucun texte dans le cœur ». La **preuve de concept** est atteignable à budget modeste ; la **preuve de gain économique** end-to-end est l'objectif le plus fragile.

## 2. Positionnement vs travaux proches

| Travail | Ce qu'il valide | Ce que MSL ajoute / exige en plus |
|---|---|---|
| **Coconut** (2412.06769) | Raisonnement en latent **continu** à l'intérieur d'un LLM texte ; meilleur trade-off accuracy/FLOPS sur logique. | Latent **discret** + **opacité** + **standard inter-modèles** + Transformer n'ayant **jamais vu** de texte (Coconut garde tokenizer + texte). |
| **Large Concept Models** (2412.08821) | Modélisation autoregressive dans un espace de représentation de phrases (SONAR, 200 langues) ; quantization optionnelle ; scaling 1.6B→7B. | Codec **appris conjointement** (LCM réutilise SONAR pré-existant) ; unité **variable** pas phrase-fixe ; canal littéral ; protocole multi-agents ; opacité revendiquée. |
| **BLT** (2412.09871) | Patches dynamiques par entropie, allocation de compute, scaling byte-level compétitif. | Allocation adaptative au niveau **sémantique** (pas byte) ; invariance de formulation ; vise bits/sens pas bytes/texte. |
| **Emergent Compositional Comm.** (2604.03266) | Protocole **discret compositionnel** émergeant par pression multi-agent (Gumbel-Softmax + iterated learning), PosDis≈0.999, généralisation, intervention causale chirurgicale. | **Le papier le plus proche du cœur MSL.** Mais limité à ~3 propriétés physiques fixes, features pré-gelées, pas de texte, pas de canal littéral, pas de standard au-delà du backbone. MSL doit étendre ça au langage + valeurs exactes. |
| **UMR/SETUP** (2512.07068) | Graphe sémantique lisible, parser English→UMR. | **Anti-exemple explicite** du brief. Sert de borne : tout ce qui ressemble à de la sémantique symbolique lisible est hors-cadre. |

**Verdict de nouveauté** : l'originalité n'est pas dans une brique, mais dans le couplage *codec discret appris + opacité + standard inter-modèles + cœur sans texte*. Aucun papier ne réalise les quatre ensemble. La revendication est donc **non triviale et non saturée**.

### Travaux additionnels à intégrer à l'audit (non cités dans le brief, pertinents)

- **VQ-VAE / VQGAN** : goulots discrets, engagement, collapse de codebook.
- **EnCodec / SoundStream / Mimi (Kyutai)** : codecs discrets sémantiques pour l'audio — modèle le plus proche d'un « codec sémantique » réel en production.
- **AudioLM / SpeechLM / GSLM** : LM natifs discrets hors texte (preuve d'existence de Transformers ayant « jamais vu de texte »).
- **dVAE (DALL·E 1)** : goulot discret + Transformer conjoint.
- **Emergent communication** (Lewis, Lazaridou, Foerster) : jeux de référence, collusion, compositionnalité — corpus théorique de H1/H2.
- **Information bottleneck** : cadre formel pour la courbe débit-distorsion.
- **Semantic communication (deep learning)** : encodage sémantique pour la transmission, cadre de la H5.

## 3. Limites informationnelles

Trois bornes théoriques encadrent ce qui est atteignable :

### Borne de compression
Un paquet de *n* codes dans *B* codebooks de taille *V* porte au plus *B·log₂V* bits d'information mutuelle avec la source. À 80 bits/paquet (config du brief), le plafond absolu est ≈10¹⁴ paquets distincts — large pour le sens, mais le budget **total** (paquets + STOP + littéraux) doit être comparé à l'entropie de la source, pas au nombre de tokens. C'est la métrique décisive et elle est bien posée dans le brief (§11).

### Borne de collusion
Théorème classique de la communication émergente : un couple émetteur/récepteur unique peut mémoriser un code privé non généralisable. Le brief le sait (§7.2, §12). La parade (multi-émetteurs/récepteurs, iterated learning) est **connue mais coûteuse** : Kaszyński 2026 a besoin de 80 seeds × 4 agents pour prouver la convergence.

### Borne de calcul de conversion
Si l'encodeur E est plus cher que le gain sur T, le bilan est négatif. BLT montre qu'on peut allouer le compute adaptativement ; MSL doit prouver que E est *amortissable* (un même encodage sert à N prédictions), sinon la « compression sans économie » du §12 se matérialise.

## 4. Les cinq hypothèses les plus risquées

Chacune est énoncée, index de risque (R ∈ {haut, moyen, bas}), puis expérience **minimale** de falsification. Les expériences sont reprises et détaillées dans `02_experiences_falsification.md`.

### H1 — Un paquet discret opaque peut porter un *état informationnel complet*, pas un concept isolé. *R = haut.*
La quasi-totalité des systèmes discrets (VQ-VAE, LCM) codent une unité localisée (image, phrase). MSL revendique un paquet = état complet distribué sur plusieurs codes. C'est l'hypothèse la plus fragile : rien ne prouve qu'un seul paquet suffise pour un contenu riche sans exploser la longueur.
*Falsification* : sur un monde synthétique à complexité croissante (1→k attributs), mesurer la longueur MSL minimale garantissant récupération parfaite. Si elle croît linéairement avec k au lieu de rester sous-linéaire, H1 est faux.

### H2 — L'apprentissage joint émerge un langage *non privé* généralisable à de nouveaux participants. *R = haut.*
C'est la condition d'existence d'un *standard* (vs un code privé). Kaszyński 2026 l'obtient pour 3 propriétés fixes ; MSL le demande pour du contenu ouvert.
*Falsification* : entraîner N émetteurs indépendants, geler leur codec, tester un récepteur naïf entraîné sur un codec différent. Si le transfert tombe au hasard, il n'y a pas de standard — seulement des codes privés compatibles intra-famille. Critère go/no-go : transfert inter-codecs > hasard + marge.

### H3 — La réduction de tokens se traduit en gain réel de FLOPS/mémoire, coût de E/Q/D inclus. *R = haut.*
Le brief le dit explicitement (§11, règle de décision). Mais l'encodeur sémantique peut être lourd ; le goulot discret ajoute des passes ; la prédiction factorisée multi-têtes ajoute des têtes.
*Falsification* : comparer, *à FLOPS d'entraînement et d'inférence égalisés*, une baseline texte et un système MSL sur même tâche. Si à FLOPS égal MSL ne dépasse pas la baseline, le « gain de tokens » est cosmétique. C'est la métrique qui tue le projet si elle échoue.

### H4 — Un Transformer entraîné *sans jamais voir de texte* reste compétitif sur des tâches sémantiques ouvertes. *R = moyen sur mondes synthétiques, haut sur langage ouvert.*
Coconut garde le texte ; LCM démarre de SONAR entraîné sur texte. Aucun système à l'échelle n'a jamais appris le « raisonnement » sans aucune exposition textuelle.
*Falsification* : après figeage du codec, comparer T_MSL (jamais vu texte) à un T_texte de même taille sur des tâches dérivées du même contenu. Sur mondes synthétiques la parité est vraisemblable ; sur documents réels, surveiller un éventuel « plafond de concept ».

### H5 — Le canal littéral peut être borné sans devenir *trou* ni détruire la compression. *R = moyen.*
Trop strict → erreurs sur valeurs exactes ; trop permissif → recopie de texte (§12). La solution de §5.4 (quotas + audit) est techniquement saine mais fragile à l'optimisation.
*Falsification* : tracer l'usage du canal en fonction du budget de bits. S'il croît proportionnellement au contenu (et non aux seules valeurs exactes), le système contourne le latent et H5 échoue par détournement.

### Bonus H6 (à surveiller) — *La compositionnalité généralise à des combinaisons absentes de l'entraînement sans dictionnaire.* *R = moyen.*
Kaszyński l'obtient en physique ; sur du langage, l'absence de dictionnaire stable peut empêcher la recombinaison. Falsification : split train/test sur combinaisons inédites, mesurer la chute vs combinaisons vues.

## 5. Tensions internes du brief (à résoudre avant le MVP)

1. **« Paquet = état complet » (§5) vs « longueur dynamique + STOP » (§5.3).** Si un seul paquet suffit pour un état complet, pourquoi une séquence ? Il faut décider si le paquet est *l'atome* (et la séquence = suite d'atomes) ou si la *séquence de paquets* est l'unité. Le brief oscille. **Recommandation** : trancher explicitement que la séquence de paquets est l'unité de message, le paquet étant l'atome — sinon H1 et la longueur dynamique se contredisent.
2. **« Aucun texte dans le cœur » (§4) vs dépendance à un encodeur E entraîné sur texte.** Le cœur n'a pas vu le texte, mais le *système* si (via E). La propriété revendiquée est réelle pour le cœur seul, mais la revendication « natif » doit être qualifiée : le standard MSL est natif, son écosystème ne l'est pas. À documenter pour ne pas survendre.
3. **« Opaque et incontrôlable » (§12) vs « tests de conformité » (§9, §11).** Un codec opaque ne se prête pas au test de conformité par inspection ; la conformité devra être *comportementale* (un émetteur conforme produit des paquets qu'un récepteur de référence décode correctement). Ce n'est pas une contradiction mais une contrainte de méthode à expliciter.
4. **Cibles chiffrées (§11.1) sans calibration.** Le brief dit lui-même que ce sont des « hypothèses de travail ». « 99 % de fidélité sur mondes synthétiques » et « 4× moins d'étapes » sont arbitraires avant la première courbe débit-distorsion. Risque : fixer des seuils qui tuent un projet qui aurait un vrai gain mais sous le seuil. **Recommandation** : reporter les seuils après Phase 1.
5. **Ordre des sources (§8.1) vs cibles de preuve (§11.1) sur « documents réels ».** Les cibles parlent déjà de « dégradation downstream », ce qui suppose du contenu réel — mais §8.1 met les documents réels en 4ᵉ position. Cohérence à clarifier : les cibles de §11.1 ne s'appliquent qu'à partir de la phase où le domaine correspondant est atteint.

## 6. Direction d'orientation

Sans anticiper l'architecture, trois axes ressortent comme prioritaires pour la suite :

- **Falsifier H1 et H2 d'abord** : ce sont les deux hypothèses dont l'échec tue le projet *structurellement* (pas seulement économiquement). H3 peut échouer sans invalider la recherche ; H1/H2 non.
- **S'inspirer du protocole Kaszyński 2026** comme squelette de la Phase 1 (Gumbel-Softmax + iterated learning + multi-seeds + intervention causale), en étendant le domaine du physique au sémantique simple.
- **Comparer systématiquement à LCM** comme baseline « concept discret pré-existant » et à **Coconut** comme baseline « latent continu » — ce sont les deux références à battre ou à distinguer.

## 7. Conclusion

Le brief est **mieux écrit que la moyenne des programmes de recherche amateurs** : il anticipe ses propres risques (§12), pose une règle de décision économique (§11), et se garde de la grammaire symbolique. Les deux angles morts sont (i) la confusion *paquet vs séquence* et (ii) des cibles chiffrées non calibrées. La revendication de nouveauté résiste à l'examen des travaux proches. **Aucune hypothèse n'est trivialement fausse**, mais H1, H2 et H3 sont *hautement* risquées et leur falsification doit précéder tout investissement en Phase 3 (Transformer natif). L'ordre des phases (§9) est correct : il **commence par falsifier**, ce qui est la bonne discipline.

---

## Références citées

- Coconut — Hao et al. 2024, arXiv:2412.06769.
- Large Concept Models — LCM team, Meta AI, arXiv:2412.08821.
- Byte Latent Transformer — Pagnoni et al. 2024, arXiv:2412.09871.
- Emergent Compositional Communication for Latent World Properties — Kaszyński 2026, arXiv:2604.03266.
- SETUP (English→UMR) — Markle, Gutierrez Bach, Wein 2025, arXiv:2512.07068.
