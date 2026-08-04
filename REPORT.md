
Audit qualité — msl_project
Prototype de recherche cohérent dans son intention, avec de vraies bonnes idées d'ingénierie (module de seeding centralisé, ground truth calculée par interpréteur, sweep reprenable). Mais j'ai trouvé deux bugs qui invalident silencieusement une grande partie des résultats et une chaîne de fuites train/test sur le pipeline vrai texte qui touche précisément les chiffres mis en avant dans le README.

1. Bugs critiques (invalident des résultats, pas juste du style)
   1.1 Les codebooks PQ/RVQ sont écrasés à chaque step
   /Users/ehouzell/Dev/projects/msl_project/src/msl/models/quantizer.py:114 et :121

self.usage.mul_(0.99).add_(counts.clamp(max=1.0), alpha=0.01)   # ligne 114
...
dead = self.usage < self.restart_threshold                       # ligne 121, threshold = 1.0
usage est une EMA bornée : point fixe = 0.01/(1-0.99) × 1 = 1.0, atteint seulement asymptotiquement. Le test usage < 1.0 est donc vrai en permanence. Conséquence, à chaque appel en mode train (quantizer.py:182-185) :

step 1 : n_dead = 256 → tous les vecteurs du codebook remplacés par des échantillons aléatoires du batch ;
steps suivants : usage[dead] = 1.0 puis ×0.99 → tout code non touché par le batch courant est réécrit à chaque step.
Le codebook n'apprend jamais : il vaut « les échantillons du dernier batch ». Cela touche tous les résultats PQ : download_hf.py:121-129, build_realtext_msl.py:51-59, test_h2.py:41-48 et :127-133, train_text_decoder.py:188-217. Le coût de 21 ms/batch attribué au quantizer dans docs/15_phase_a_complete.md s'explique aussi par les 48 cdist + 48 restarts complets par step.

Note secondaire : restart_dead est appelé avec un batch de 16 dans train_text_decoder.py (batch_size=16, ligne 194) — 256 codes remplis depuis 16 vecteurs.

1.2 La vérité terrain MS-1 n'est pas reproductible entre processus
/Users/ehouzell/Dev/projects/msl_project/src/msl/data/ms1.py:237

new = domain[(hash((p["action"],)) % len(domain))]
hash() sur un tuple de str est salé par PYTHONHASHSEED, qui doit être fixé avant le démarrage de l'interpréteur. seeding.py:47 le pose après → sans effet. Vérifié : 3 processus donnent 3, 0, 3. Donc les labels des tâches composition (1/5 des tâches générées par sample_balanced_tasks, ms1.py:434-435) changent d'un run à l'autre. Cela contredit directement le docstring ms1.py:3 (« fully deterministic given a seed ») et AGENTS.md. Aucun test ne couvre solve(composition).

1.3 L'éval round-trip vrai texte utilise un quantizer différent de celui de l'entraînement
/Users/ehouzell/Dev/projects/msl_project/src/msl/eval/test_roundtrip_realtext.py:28-47

Le checkpoint contient bien le quantizer entraîné (train_text_decoder.py:258), et il est ignoré : le script en ré-entraîne un neuf (lignes 39-47). Le décodeur GPT-2 reçoit donc des z_q issus d'un codebook qu'il n'a jamais vu. Lignes 40-41 : quantizer.eval() immédiatement suivi de quantizer.train() — code mort résiduel. Comparer avec end_to_end.py:79-91 qui, lui, charge correctement le quantizer du checkpoint : les deux évals ne mesurent pas le même système.

2. Solidité méthodologique
   2.1 Fuites train/test — systématiques sur le pipeline vrai texte
   Endroit	Nature
   train_native_lm.py:82-83	loader et test_loader pointent le même ds. Le per_code_acc sauvé ligne 141 est une métrique train.
   train_text_lm.py:123	éval finale sur loader (train). Aucun test set.
   test_roundtrip_realtext.py:62-64	commentaire honnête : # unseen during training if we split. Or TextDataset (train_text_decoder.py:127) prend tout le corpus → les 20 phrases « test » sont des phrases d'entraînement.
   test_h2.py:127-196	le « standard » et le « nouvel encodeur » sont entraînés sur embeddings[:5000], l'agreement est mesuré sur embeddings[:1000] — sous-ensemble du train.
   train_semantic_quantizer.py:88-118	entraînement puis quantification/retrieval sur le même corpus complet.
   Le chiffre phare 93,7 % de docs/15_phase_a_complete.md (§H1) est donc une métrique train. Le côté synthétique n'échappe pas : build_msl_corpus.py:45 construit le corpus avec seed_floor=0 (graines train) et train_native_lm l'évalue sur lui-même.

À l'inverse, le pipeline synthétique du codec est propre : train_codec.py:196-200 et train_text_baseline.py:114-117 construisent un vrai test set avec seed_floor=2_000_000. Bon point.

Nuance sur la convention anti-fuite : PrecomputedMS1Dataset (dataloader.py:103) génère des graines dans [seed_floor, seed_floor + 999_999], donc [2_000_000, 2_999_999] en test — hors du TEST_RANGE = (2_000_000, 2_010_000) déclaré (ms1.py:466-470). Pas de fuite réelle (disjoint du train), mais la convention n'est appliquée nulle part par le code, seulement par des constantes.

2.2 La tâche du LLM natif MSL est structurellement non apprenable
/Users/ehouzell/Dev/projects/msl_project/src/msl/data/build_realtext_msl.py:74-77

# Each sequence = seq_len consecutive sentences' packets.

sequences = all_packets[:n_sequences*seq_len].reshape(n_sequences, seq_len, n_codebooks)
Le corpus est un dump Tatoeba dédupliqué (download_hf.py:89-93) : des phrases sans aucun lien entre elles. Le « LLM natif » est donc entraîné à prédire le paquet de la phrase n+1 à partir de phrases sans rapport. Un per_code_acc de 93,7 % dans ce cadre ne mesure pas une capacité de modélisation : combiné au bug 1.1 (codebooks dégénérés) et à la fuite 2.1, il mesure très probablement le mode marginal des codes. docs/15 en fait le pilier de « H1 validé sur vrai texte ».

2.3 Comparaison baseline vs MSL : non équitable, et dans les deux sens
Asymétrie d'information (synthétique). TextBaseline.forward reçoit task_ids = tokenisation de f"{t.kind} {t.payload}" (dataloader.py:121), donc eid, key, value de la question. Codec.forward ne reçoit que task_kind_id ∈ [0,4] (codec.py:102-103, choix documenté et assumé lignes 99-101). Pour une tâche qa, le codec ne sait pas quel attribut de quelle entité est interrogé — il ne peut structurellement pas répondre mieux que le mode conditionnel de l'état. Les task_acc des deux modèles ne sont pas comparables, et le « gap » H1 mesure une capacité dégradée par construction.

Ablation bruit hors distribution. train_codec.py:66-68 :

noise = torch.randn_like(z_q)
dec = codec.decoder(noise, batch["task_kind_id"])
PQQuantizer._quantize normalise z (quantizer.py:167), donc z_q a des sous-vecteurs de norme bornée. Un randn brut est très loin de la variété → le décodeur reçoit du bruit reconnaissable comme tel. Le contrôle honnête serait un shuffle de z_q dans le batch (préserve la marginale, détruit l'appariement). Le « gap noise » (métrique centrale de docs/05) est donc majoré par construction.

Comparaison de vitesse pilotée par deux constantes en dur. end_to_end.py:177-178 :

msl_pipeline_ms  = enc_ms + quant_ms + msl_step_ms * 10   # 10 packets
text_pipeline_ms = text_step_ms * 40                      # ~40 tokens per sentence
Le ratio 40/10 = 4 est le résultat ; rien ne le mesure. Pire, les deux *_step_ms ne sont pas des « steps » : msl_lm_step fait un forward sur (64, 10, 48) (10 positions) et text_lm_step un forward sur (64, ≤48) (48 positions, docs/07:144 l'admet : séquence paddée à 96). On multiplie ensuite chacun par 10 et 40. Le head texte produit (64,48,50257) vs (64,10,48,256) côté MSL — deux ordres de grandeur d'écart sur le seul tenseur de logits.

Ajouts : end_to_end.py:158 mesure le LLM MSL sur des paquets aléatoires (torch.randint), pas les vrais ; time_fn (lignes 46-54) synchronise CUDA mais pas MPS → tout chiffre produit sur Apple Silicon est du bruit ; la section MÉMOIRE (lignes 205-221) retourne 0 hors CUDA, donc les « 10 % moins de mémoire » du README ne peuvent pas venir de ce code sur Mac ; les lignes 197-199 affichent "Text: ~7 tokens per sentence" / "Ratio: 7x fewer steps" en texte pur, en contradiction avec le 40 de la ligne 178.

Le modèle texte de end_to_end.py:113-117 n'est jamais entraîné (acceptable pour du timing), mais du coup la section QUALITY (ligne 236) ne rapporte que MSL : aucune comparaison de qualité n'existe.

2.4 Métriques auto-réalisatrices
test_roundtrip.py:46-72 et structured_codec.py:153-224

Les faits sont extraits de la source et de la génération par le même parseur regex lossy. Toute paraphrase que le parseur ne sait pas lire (ex. le template FR "{e} : {key} vaut {v}", ms1.py:~340, qui n'a pas de = avant la valeur, alors que la regex test_roundtrip.py:57 l'exige) disparaît du numérateur et du dénominateur. Le score ne peut que monter. Idem pour le seuil sim > 0.5 en dur (test_roundtrip_realtext.py:94) sur des embeddings MiniLM mean-pooled non normalisés, sans baseline aléatoire — deux phrases quelconques dépassent souvent 0.5.

2.5 « 100 % de fidélité » = la fonction identité
structured_codec.py:58-150 + docs/09_traducteur_fonctionne.md

state_to_packets / packets_to_state sont deux fonctions pures inverses par construction — le doc le dit d'ailleurs franchement (« Pas d'apprentissage, pas d'erreur »). Le 159/159, 20/20 mesure donc une table de correspondance écrite à la main, pas un codec appris. Et ce n'est même pas inversible en général : les caps sont 12 attributs / 6 relations / 6 événements (structured_codec.py:50-55) avec troncature silencieuse ([:ATTR_END - ATTR_START], ligne 68), alors que le générateur va jusqu'à k=64. Le test et le Quick start du README utilisent k=8, sous le seuil. Le docstring lignes 7-16 décrit d'ailleurs un tout autre layout (slots 8-71 / 72-103 / 104-135) que le code.

2.6 Cherry-picking README vs docs
Les docs internes sont souvent honnêtes ; le README ne retient que la meilleure moitié.

README	Source réelle
« H1 … 4/5 faithful paraphrases (real text) »	docs/11:81-82 : « Le test utilise l'embedding continu (z_q avant quantization), pas les paquets discrets. » Avec quantization : docs/15 dit 2/5. Le chiffre avancé pour prouver que les paquets portent le sens a été obtenu sans paquets.
« 6× faster inference, 10 % less memory »	Chiffre synthétique (docs/07:127). Sur vrai texte : docs/15 dit 3,7× et 17 %.
« with faithful round-trip on real text »	Voir 2.1 (éval sur données d'entraînement) + 1.3 (mauvais quantizer).
« Qualité comparable : 71 % codes vs 68 % tokens » (docs/07:26)	71 % de codes justes sur codebook de 1024 vs 68 % de tokens sur vocab ~200 : espaces d'étiquettes et baselines de hasard incomparables.
« 22 tests, 91 % coverage »	Le .coverage ne contient que 6 modules sur 25 : ms1, codec, encoder, quantizer, tokenizer, seeding. train/, eval/, data/dataloader.py, codec_v2, state_decoder, nar_decoder, native_lm, structured_codec, text_baseline sont à 0 % et hors périmètre du chiffre.
3. Les tests
22 tests, tous CPU, rapides, sans réseau — conformes à AGENTS.md. Mais ils couvrent le générateur et les shapes, jamais la logique qui produit les résultats.

Ce qui est réellement testé : tailles de vocabulaire (test_ms1.py:24), déterminisme du générateur et du renderer, difficulty == k, solve pour qa/implication/contradiction, shapes et bits des 3 quantizers, flux de gradient à travers le straight-through (test_quantizers.py:45-52), un overfit forward/backward du codec (test_codec.py:81-98).

Tests superficiels ou vacuous :

test_quantizers.py:66-75 — le seul test censé détecter le collapse du codebook n'assert que active_div > 0.0. Vrai par construction. Il ne détecte pas le bug 1.1.
test_ms1.py:137-139 — test_test_seeds_are_isolated_from_train n'assert que TEST_RANGE[0] > TRAIN_RANGE[1], deux constantes. Il ne teste pas la dérivation de graines de dataloader.py:103, seul endroit où une fuite pourrait se produire.
test_ms1.py:82-95 — assert solve(t, s) == a.value or solve(t, s) is None : la disjonction rend l'assertion presque toujours vraie.
test_codec.py:97-98 — final_loss < 1.0 sur un batch de 1 état après 200 steps : seuil arbitraire.
Trous majeurs : solve(composition) (donc le bug 1.2 passe), sample_balanced_tasks (l'équilibrage des baselines par type de tâche n'est jamais vérifié — c'est pourtant l'argument anti-biais du projet), structured_codec round-trip (annoncé à 100 % dans le README, testé nulle part, cassé au-delà de 12 attributs), PrecomputedMS1Dataset vs MS1Dataset (la docstring dataloader.py:73 affirme « Bit-identical to MS1Dataset (same seed derivation) » — c'est faux, _state_seed ajoute epoch * 104729 et rend deux langues différemment, aucun test ne le vérifie), l'ablation bruit, native_lm.loss, state_decoder, text_baseline.

4. Red flags — code mort, duplication, chiffres en dur
   Le mécanisme du baseline est mort. text_baseline.py:45,57,60 : self.task_query est créé, q calculé, puis q = q.expand(...) + task_h * 0.0 — un no-op, et q n'est jamais utilisé (ligne 61 passe task_h comme requête). Le paramètre task_query ne reçoit aucun gradient. C'est du code mort dans le modèle de référence.
   end_to_end.py:66 puis :74 : n_codebooks assigné deux fois, la première est morte. end_to_end.py:155 : test_packets calculé, jamais utilisé.
   test_h2.py:154-174 : bloc dupliqué — z_new, q_out, new_packets, new_z_q calculés (lignes 154-160), puis recalculés à l'identique (165-172). La première moitié est du calcul jeté.
   test_h2.py:96-97 : torch.eye(2000, ...) en dur alors que la taille vient du corpus. Baseline de hasard incohérente entre test_h2.py:92 (10/2000) et :235-236 (0.05 %, 0.5 %).
   configs/a10_full.yaml : référencé nulle part (vérifié sur src, docs, README). Il déclare freeze_gpt2: false alors que le code gèle tout sauf les 2 dernières couches (train_text_decoder.py:177-179), et eval: n_samples: 20 # held-out alors qu'aucun split n'existe.
   structured_codec.py:41-43 : ATTR_SLOTS=64, REL_SLOTS=32, EVENT_SLOTS=32 définis puis contredits par les constantes réellement utilisées lignes 50-55.
   ms1.py:207 : section # --- Sampling --- vide. train_native_lm.py:26 : paramètre split ignoré. train_state_decoder.py:32 : self.renderer = MS1 # just the class.
   Bits calculés en dur avec log2(256) implicite : n_codebooks*8 dans download_hf.py:150, build_realtext_msl.py:47, train_text_decoder.py:190, end_to_end.py:92 — faux dès que codebook_size ≠ 256, alors que Quantizer.bits_per_packet existe.
   Duplication de boucles d'entraînement : le même squelette (cosine LR inline, running dict, clip 1.0, éval, save) est recopié dans train_codec.py, train_codec_v2.py, train_text_lm.py, train_native_lm.py, train_text_baseline.py, train_text_decoder.py, train_state_decoder.py. cosine_lr existe 3 fois (train_codec.py:45, train_codec_v2.py:39, inline dans train_native_lm.py:99 et train_text_lm.py:95 avec 3.14159 en dur).
5. Qualité générale — points positifs et frictions
   Bon : utils/seeding.py (dérivation de sous-graines par backend, SeedBundle frozen) est propre et au-dessus de la moyenne des protos ML. Ground truth MS-1 calculée par interpréteur plutôt qu'annotée (ms1.py:214-262) — excellent choix de design. sample_balanced_tasks (ms1.py:311-...) traite explicitement le problème des marginales par type de question. sweep_h1.py:86-106 est reprenable après interruption. Typage assez systématique (from __future__ import annotations, dataclasses, retours annotés), docstrings concises, pyproject.toml avec ruff+mypy configurés.

Frictions :

Device : default_device() (seeding.py:50-55) gère CUDA→MPS→CPU correctement, mais 4 fichiers le contournent avec un mps or cpu en dur qui exclut CUDA : test_h2.py:32 et :120, train_semantic_quantizer.py:71, build_realtext_corpus.py:~50. docs/15 affirme avoir lancé test_h2 sur A10 : ce script y a tourné sur CPU. test_roundtrip_realtext.py:23-24 réimplémente la cascade à la main. Le README dit MPS ou CUDA, le code dit parfois MPS seul.
Reproductibilité : pas de torch.use_deterministic_algorithms, pas de cudnn.deterministic, PYTHONHASHSEED posé trop tard (cf. 1.2), et de nombreux np.random.randint sur le RNG legacy global (test_h2.py:44, download_hf.py:124, train_semantic_quantizer.py:89) alors que AGENTS.md impose de passer par msl.utils.seeding.
Gestion d'erreurs : quasi absente là où ça compte. torch.load(..., weights_only=False) partout sans try/except ni vérification de présence des clés (end_to_end.py:63,72,95 échouera par KeyError/FileNotFoundError brut). except Exception as e: print(...); return [] (download_hf.py:71-73) avale silencieusement l'échec Wikipedia → un corpus « 100k » peut n'être que du Tatoeba sans qu'aucun log ne l'indique en aval.
train_text_decoder.py:222 : clip_grad_norm_(decoder.proj.parameters(), 1.0) ne clippe que la projection alors que l'optimiseur met aussi à jour les 2 dernières couches GPT-2 + le lm_head (lignes 177-182, 200).
train_text_decoder.py:136 + :89-92 : padding avec eos_token_id puis cross-entropy sans ignore_index → la loss est dominée par la prédiction du padding. codec_v2.py:126 et nar_decoder.py:115 utilisent ignore_index=0 correctement ; incohérence entre les modules.
dataloader.py:141-142 : commentaire qui décrit une correction… suivi de rien. La ligne 153 fait 1 - self.recon_lang_id[idx] en mode same-lang, ce qui suppose implicitement 2 langues alors que n_langs=3 (encoder.py:78) — lang_id=2 (structuré) est inatteignable.
6. Priorités si l'objectif est de rendre les conclusions défendables
quantizer.py:121 — corriger la condition de dead-code (ex. seuil à 1e-3 sur une EMA de fréquence, ou compteur de steps depuis dernier usage). Puis relancer tout ce qui touche PQ. Sans ça, aucun résultat quantizer ne signifie quelque chose.
ms1.py:237 — remplacer hash() par une fonction déterministe (ACTIONS.index(action) ou un hashlib stable).
Introduire un split réel sur le corpus vrai texte (indices train/val/test sauvés dans big_corpus.pt) et l'utiliser dans train_text_decoder, train_native_lm, train_text_lm, test_roundtrip_realtext, test_h2.
test_roundtrip_realtext.py:39-47 — charger dec_ckpt["quantizer"] au lieu de réentraîner.
end_to_end.py:158,177-178,199 — mesurer le nombre réel de tokens/paquets par phrase, un vrai décodage pas-à-pas avec KV-cache des deux côtés, torch.mps.synchronize(), et un budget mémoire contrôlé (même taille de vocab de sortie ou mesure hors logits).
train_codec.py:67 — remplacer randn_like par un shuffle de z_q dans le batch.
Aligner l'information donnée au codec et au baseline (soit les deux voient la question, soit aucun), sinon retirer la comparaison des conclusions.
Aligner le README sur les docs : 3,7× (pas 6×), 2/5 avec quantization (pas 4/5 sans), et préciser le périmètre du 91 % de couverture.
J'ai terminé la revue complète (docs + code + logs + artefacts), sans rien modifier. Voici mon verdict.

TL;DR
L'idée est pertinente et la démarche (auditer → falsifier → mesurer) est la bonne. Mais en l'état, aucune des trois validations « H1/H2/H3 sur vrai texte » ne tiendrait face à un examen externe : un bug silencieux invalide tous les résultats du quantizer, les évaluations se font sur les données d'entraînement, et les comparaisons de vitesse reposent sur des constantes codées en dur plutôt que sur des mesures. La bonne nouvelle : c'est réparable, et l'infrastructure pour le faire existe déjà.

Ce qui est de qualité
docs/01_audit_critique.md est un excellent document : littérature correcte (Coconut, LCM, BLT, communication émergente), risques bien identifiés, et surtout des critères de falsification stricts et bien posés. Le problème n'est pas ce document — c'est que l'implémentation ne respecte pas ses propres critères (j'y reviens).
La discipline documentaire (15 docs séquentiels, verdicts intermédiaires, limites auto-signalées) est au-dessus de la moyenne. docs/07_verdict_economique.md fait même 3 seeds et inclut le coût du codec.
L'ingénierie est propre pour un proto : seeding centralisé, ground truth MS-1 calculée par interpréteur, typage systématique, ruff/mypy/pytest, runs/ correctement hors git. src/msl/utils/seeding.py et src/msl/data/ms1.py sont de bons modules.
Ce qui ne tient pas

1. Un bug invalide tous les résultats quantizer. Dans quantizer.py:114-121, usage est une EMA dont le point fixe est exactement 1.0… qui est aussi le restart_threshold. Le test usage < 1.0 est donc quasi toujours vrai : au premier step, les 256 codes de chaque codebook sont écrasés par des échantillons du batch, puis à chaque step tout code absent du batch courant est réécrit. Le codebook n'apprend jamais — il vaut « les derniers échantillons vus ». Tous les résultats PQ (H1, H2, H3, décodeur) sont construits sur ce quantizer dégénéré. Ça explique d'ailleurs les 21 ms/batch du quantizer notés comme mystérieusement chers dans doc 15.
2. H1 (93,7 %) est une métrique d'entraînement sur une tâche non apprenable. Dans train_native_lm.py:81-83, loader et test_loader pointent le même dataset — pas de split, le 93,7 % est mesuré sur le train. Et dans build_realtext_msl.py:74-77, les « séquences » sont 10 phrases Tatoeba consécutives sans aucun lien sémantique entre elles : prédire « le paquet de la phrase suivante » n'a pas de signal à capter. Le 93,7 % mesure très probablement la distribution marginale des codes (aggravée par le bug n°1), pas une capacité à « penser en MSL ». Aucune baseline triviale (prédire le code majoritaire) n'est calculée nulle part.
3. H2 ne teste pas l'adoption d'un standard. Le « nouvel encodeur » de test_h2.py:148 est un simple Linear(384, 384) posé sur les mêmes embeddings MiniLM, entraîné sur les mêmes phrases, évalué sur un sous-ensemble de son propre train. Que 87 % soit atteignable montre qu'une application linéaire converge vers ~l'identité — pas qu'un modèle indépendant peut adopter le codec. Ton propre audit (doc 01, H2) exigeait « N émetteurs indépendants, récepteur naïf, codec différent » : c'est beaucoup plus fort que ce qui a été implémenté. Le test « émergence » mesure surtout la stabilité du k-means sur les mêmes données.
4. H3 (3,7×–6×) est une hypothèse, pas une mesure. Dans end_to_end.py:177-178, le ratio vient de deux constantes en dur (× 10 paquets vs × 40 tokens) — le résultat est écrit dans le code. La baseline texte n'est jamais entraînée, donc l'équivalence de qualité est supposée alors que le décodeur réel produit 2/5 paraphrases fidèles. Le coût du décodage GPT-2 (124M) est exclu du pipeline MSL. Et time_fn synchronise CUDA mais pas MPS — tout chiffre produit sur Mac est du bruit. L'audit initial exigeait une comparaison « à FLOPS égalisés, même tâche » : ce n'est pas ce que fait ce script.
5. Le README survend par rapport aux docs. « 4/5 paraphrases fidèles » vient d'un test sans quantization (doc 11 le dit explicitement ; avec quantization c'est 2/5). « 6× plus rapide » est le chiffre synthétique (vrai texte : 3,7×, lui-même douteux cf. point 4). « 91 % coverage » ne couvre que 6 modules sur 25 — tout train/ et eval/ (là où se joue la validité scientifique) est à 0 %.

À noter aussi : hash() non seedé dans ms1.py:237 rend les labels des tâches « composition » non reproductibles entre processus, et test_roundtrip_realtext.py:39-47 ré-entraîne un quantizer neuf au lieu de charger celui du checkpoint — le décodeur est évalué sur des codes qu'il n'a jamais vus.

Est-on sur la bonne voie ?
Sur la direction : oui. Le sujet est d'actualité, le positionnement vs LCM/Coconut est correct, et l'ordre des phases (falsifier avant d'investir) est le bon. Ce qui a été réellement démontré à ce stade est plus modeste que les claims : quantizer des embeddings MiniLM en 384 bits préserve une partie utile de leur structure sémantique, et GPT-2 peut être conditionné pour reconstruire partiellement le texte. C'est un vrai résultat de départ — mais la sémantique vient de MiniLM pré-entraîné, pas d'un langage machine émergent.

Sur l'exécution : il faut un arrêt qualité avant la Phase B. Passer à BGE-M3 maintenant reviendrait à empiler du travail sur des mesures invalides. Dans l'ordre :

Corriger le bug du dead-code restart (quantizer.py:121) et relancer tout ce qui touche PQ — sans ça, rien d'autre n'a de sens.
Split train/val/test réel sur le corpus texte, utilisé partout (décodeur, LLM natif, H2, round-trip).
Baselines triviales : prédicteur du code majoritaire pour le LLM natif, paires de phrases aléatoires pour le seuil de similarité, permutation pour les métriques NN de H2.
Refaire H3 honnêtement : baseline entraînée, longueurs mesurées (pas 40/10 en dur), décodage réel pas-à-pas, mps.synchronize(), coût du décodeur inclus ou tâche machine-à-machine définie symétriquement.
Refaire H2 avec un vrai second encodeur — la Phase B (BGE-M3) est justement l'occasion parfaite : si BGE-M3 peut adopter le codec figé de MiniLM, là tu auras un vrai résultat d'adoption.
Corpus séquentiel avec du signal : des paragraphes Wikipédia dans l'ordre plutôt que des phrases Tatoeba indépendantes, sinon « prédire le paquet suivant » ne veut rien dire.
Aligner le README sur les docs (3,7×, 2/5 avec quantization, périmètre réel de la couverture).
Le projet a exactement le bon squelette — audit lucide, falsification d'abord, documentation honnête dans les docs internes. Ce qui manque, c'est que la rigueur du doc 01 redescende dans le code d'évaluation. Si tu veux, je peux appliquer les corrections 1 à 4 (elles sont mécaniques et bien délimitées) — mais je n'ai touché à rien, comme demandé.
