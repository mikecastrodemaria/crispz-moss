# crispz-moss : cahier des charges pour Claude Code

`CLAUDE.md` du repo `mikecastrodemaria/crispz-moss`. Claude Code le lit à l'ouverture du
projet. Rédigé le 03/11/2026 depuis crispz-ideogram 0.1.0, à partir du code publié de MOSS-Image (pipeline
`pipeline_moss_image.py` lu en entier), de son guide utilisateur, du rapport technique
(arXiv 2610.23374) et de trois tests indépendants. Même méthode que `CLAUDE.md` de
crispz-ideogram : lots courts, validation de Mike entre chaque lot, mesures sur la 5090.

## 1. Objectif

Un studio local autour de **MOSS-Image** (OpenMOSS, Fudan, sorti le 11/10/2026) : le
frère **commercialisable** de crispz-ideogram. Même positionnement (texte dans l'image,
affiches, logos, packaging, slides, typographie dense, contrôle par légende JSON avec
boîtes), même interface, mais des poids **Apache 2.0** : un livrable client produit en
local est permis, là où Ideogram 4 l'interdit sans licence négociée.

Ce que MOSS apporte en plus d'Ideogram 4 : licence libre, modèle deux fois plus petit
(5B contre 2 x 9,3B), prompt négatif réel, un seul transformer, LoRAs entraînables dès
6 Go (KohakuMossTrainer, DiffSynth), un écosystème déjà vivant (distillations 4 et 8
steps, LoRA d'édition avec références, LoRA anti-banding, repack ComfyUI, GGUF, MLX).
Ce qu'il n'apporte pas, ou pas encore : la preuve, sur nos affiches, que le texte est
aussi fiable que celui d'Ideogram 4 (les tests indépendants se contredisent, voir §2).
**Le lot 1 tranche cette question par un A/B avant tout autre travail.**

## 2. Le modèle : ce qui est vérifié (03/11/2026)

| | MOSS-Image (local) |
|---|---|
| Poids | `OpenMOSS-Team/MOSS-Image-SFT` (usage général) et `MOSS-Image-Base` (base des LoRAs), non gated |
| Licence | Apache 2.0, poids et code de pipeline ; encodeur Qwen3.5-9B Apache 2.0 ; VAE repris de FLUX.2 klein 4B (Apache 2.0, à confirmer au lot 6) |
| DiT | 5B, 42 blocs, dim 3072, 24 têtes (head_dim 128), FFN 8192, SiTU-GLU, RoPE 4D, modulation partagée |
| Encodeur | Qwen3.5-9B gelé (`Qwen3_5Model`, 32 couches, hidden 4096), `last_hidden_state` seulement, tour vision incluse dans les poids (`qwen3_5_vision`) |
| VAE | `AutoencoderKLFlux2` (32 canaux, normalisation `bn`), le même que FLUX.2, FLUX.3 et Ideogram 4 |
| Latents | 128 canaux sur grille /16 (2 x 2 patches de 32 canaux), rangés dans l'ordre FLUX `(c, p, p)` |
| Taille des fichiers (bf16) | DiT 10,2 Go + encodeur 16,8 Go + VAE 0,25 Go = **27,2 Go** |
| Réglages officiels | 50 steps, CFG 3,0 (3 à 6 conseillé), shift 5,0, Euler, `uncond_skip_layers=[22]`, négatif vide ou `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` |
| Résolution | multiples de 16 ; recommandé 1, 1,5 ou 2 Mpx ; extrapole de 0,25 à 4 Mpx ; ratio libre |
| Prompts | texte libre (anglais conseillé, chinois ok), séquences de tags, layout JSON ; `\n` pour forcer les retours à la ligne du texte rendu |

Le pipeline, tel que publié (`custom_pipeline="pipeline_moss_image"`, `trust_remote_code`,
440 lignes, Apache 2.0) :
- Classes : `MOSSImageTransformer`, `MOSSImageScheduler` (sous-classe de
  `FlowMatchEulerDiscreteScheduler` : sigmas décalées par `shift` puis **alignées sur la
  grille d'entraînement**, doublons retirés ; `set_timesteps(num_inference_steps)` seul,
  pas de paramètre `sigmas`), `MOSSImagePipeline(DiffusionPipeline)`.
- **Absents** : `callback_on_step_end`, `_interrupt`, mixin LoRA, paramètre `sigmas` ou
  `timesteps`. Présents : `latents`, `prompt_embeds` + `prompt_attention_mask`,
  `output_type="latent"`, `num_images_per_prompt`, `uncond_skip_layers`, SLG optionnel
  (`slg_layers`, `slg_scale`, `slg_start`, `slg_stop`).
- Le négatif passe par l'encodeur (deux encodages par image quand il change) ; par
  défaut CFG = deux passes de transformer par step, la branche négative sautant le bloc 22.
- Le transformer **boucle sur le batch** (une passe par image, à la vraie longueur de
  texte, aucun masque dans SDPA) : batch N = N passes, pas d'attention gonflée par le
  rembourrage (le problème corrigé sur Ideogram n'existe pas ici). head_dim 128 : le
  noyau flash de SDPA est éligible (Ideogram, à 256, n'avait que `efficient`).
- Encodage du prompt : gabarit de chat Qwen (`add_generation_prompt`,
  `enable_thinking=True`), 4096 tokens max, un prompt à la fois sans rembourrage.
- Bruit initial et pas d'Euler en fp32, DiT et encodeur en bf16, VAE en fp32.
- Décodage : dénormalisation `bn` sur les 128 canaux tels que stockés, puis
  `(b, c//4, 2, 2, h, w)` vers `(b, c//4, 2h, 2w)`.
- `model_cpu_offload_seq = text_encoder->transformer->vae`, `maybe_free_model_hooks()`
  en fin d'appel.

Mesures publiées par des tiers, à refaire chez nous : RTX 5090, ComfyUI, DiT et encodeur
en int8, 1088x1920, 50 steps : **37 s par image, 15,5 Go** ; checkpoint DMD 8 steps à
CFG 1 : 3,9 s ; LoRA 4 steps : 2,2 s. RTX 4070 12 Go, int8, 832x1216 : 52 s. Diffusers
bf16 sans quantification à 2048x1152 : plus de 32 Go. Chez OpenMOSS, 1024x1024 en
50 steps avec leur profil `minimal` (DiT en RAM) : 3,65 Go de VRAM, 14 Go de RAM.

Qualité du texte, état des sources : le rapport annonce 96,3 sur son propre benchmark
LLTB (Qwen-Image 2512 : 64,9) et un mIoU de boîtes de 0,88 contre 0,44 pour Ideogram 4 ;
son classement humain (2e derrière GPT Image 2) compte un prompt enhancer GPT-6. Tests
indépendants : MindStudio juge le texte et les boîtes « hésitants, pas prêts pour la
prod » et le photo-réalisme excellent ; ai-muninn obtient 4 titres chinois sur 4 contre
0 sur 4 pour Qwen-Image 2.1, mais rate les petits textes fixes, et constate que les
variantes 4 et 8 steps ajoutent de faux textes anglais en fond. Défaut connu du SFT :
**banding vertical** (LoRA communautaire DeBand). Variante FP16 corrigée existante,
inutile en bf16.

À vérifier au lot 1 (non confirmé) : quantification nf4 bitsandbytes de l'encodeur
Qwen3.5 (couches GDN hybrides, noyaux `flash-linear-attention` absents sous Windows :
repli torch, vitesse d'encodage à mesurer) ; RAM nécessaire à l'offload « encodeur
seul » (16,8 Go en RAM) ; vitesse réelle en bf16 sans quantification du DiT.

**Fichiers uniques Hugging Face et CivitAI (demande de Mike, 03/11/2026)** : la
compatibilité avec les poids publiés hors du repo diffusers fait partie du lot 1.
Relevé (en-têtes safetensors lus le 03/11/2026) :
- `Comfy-Org/MOSS-Image` (HF) et la page CivitAI « MOSS Image » (miroir officiel, 3005406)
  publient le MÊME format ComfyUI : DiT bf16 (9,93 Go, 379 tenseurs) ou
  `int8_tensorwise` + ConvRot (5,29 Go : poids int8, `weight_scale` par ligne, blobs
  `comfy_quant` et `comfy_attention.config`) ; encodeur `qwen3.5_9b` en bf16 (19,3 Go,
  clés `model.language_model.*`, `model.visual.*`, `lm_head`, tête `mtp`), int8 ConvRot
  (9,75 Go) ou `asym_w4a8_int8` GPTQ (7,34 Go, codebook 16 valeurs) ; VAE `flux2-vae`.
- La seule différence de clés avec le repo diffusers : ComfyUI fusionne les projections
  d'attention du DiT en `layers.N.proj.qkv` [9216, 3072] (q, k, v dans cet ordre) ; les
  blocs texte ont déjà `qkv` fusionné en amont. `rope.bands` est dans les deux.
- « Dirty-Moss » (CivitAI 3015644, fusion de LoRAs NSFW par Sateluco) : DiT seul, même
  format (bf16 9,93 Go ou int8 5,29 Go). Se charge comme fichier unique de DiT.
- ConvRot = rotation de Hadamard par groupes de 256 (64 sur la tour vision), matrice
  « regular hadamard » de comfy-kitchen (base H4, produits de Kronecker, PAS Sylvester),
  déjà écrite dans `cz_pipeline._hadamard_ortho` pour FLUX.3 ; `cz_moss` la reprend.
- Pages communautaires (recherche API CivitAI, 03/11/2026) : la plupart ne publient que
  l'INT8 ConvRot (5,29 Go), parfois bf16 (9,93 Go), fp8 (cast brut ou « scaled »), INT6,
  INT4 et Q4 (GGUF). Ce qui se charge ici : bf16, fp8, INT8 ; refusés par nom : INT6,
  INT4 / w4a8, Q4 GGUF, MXFP8. Réponse à Mike : prendre l'INT8 (ou le bf16 s'il existe).
- Décision : le DiT bf16, fp8 et int8 ConvRot et l'encodeur bf16 et int8 ConvRot sont lus
  (déquantifiés en bf16 en RAM, `qkv` scindé, préfixe `model.` retiré, `lm_head` et `mtp`
  jetés) ; les formats 4 bits (`asym_w4a8_int8`, `convrot_w4a4`, nvfp4) et 6 bits sont
  refusés PAR LEUR NOM dans le menu, avec l'alternative (le fichier bf16 ou int8 de la
  même page). Les LoRAs CivitAI (`lora_A`/`lora_B` sans `.weight`, métadonnées
  `moss-edit-lora` ou `hayate_lora_format`) : lot 4.

## 3. Base : crispz-ideogram, pas crispz-flux3

| Candidat | Pour | Contre |
|---|---|---|
| **crispz-ideogram 0.1.0** (recommandé) | éditeur de légende avec boîtes à la souris, presets de design, magic prompt (local / Ollama), Describe local, SDEdit et inpaint émulés sur ce même VAE, offload « encodeur seul » déjà écrit pour un encodeur Qwen de 8 à 9B, garde VRAM, handoff vers crispz-flux3, pile diffusers 0.42.1 + transformers 5.21 déjà compatible avec MOSS | schéma JSON à traduire (0..1000 contre pixels centrés) ; reste de code FLUX.3 et Ideogram à retirer |
| crispz-flux3 | même VAE, même pile | aucun éditeur de légende, aucun preset de design : tout le lot 2 d'Ideogram serait à refaire |

**Décision : forker crispz-ideogram `main` (0.1.0).** Remotes : `origin` = crispz-moss,
`ideogram` = crispz-ideogram (pour reprendre les correctifs communs).

Ce qu'on garde : toute l'app, `cz_caption` / `cz_caption_ui` / `cz_design` / `cz_magic`,
`cz_vector`, `cz_handoff`, le détailleur, les tuiles, l'ESRGAN, la queue, le XYZ, le
protocole v1, les tests. Ce qu'on remplace : le moteur (`cz_ideogram.py` devient
`cz_moss.py`), les profils, le schéma de légende, les presets de steps. Ce qu'on retire :
`cz_ideogram_api.py` et `cz_cloud_ui.py` (MOSS n'a aucun cloud, tant mieux),
`third_party/ideogram4`, le code FLUX.3 de `cz_pipeline.py` resté pour les fusions.

## 4. Règles de travail

- Commits au nom de Mike (`Mike Castro Demaria <supersoniquestudio@gmail.com>`), auteur
  et committer, sans aucune ligne d'attribution à Claude.
- Ni tiret cadratin ni emoji, nulle part.
- Étape par étape : à la fin de chaque lot, s'arrêter, résumer, attendre la validation.
- Aucune fonction n'envoie de données hors de la machine ; Ollama compris (serveur local
  seulement, URL affichée dans le log à chaque appel).
- Toute dégradation est annoncée dans le log (`_log`), jamais silencieuse.
- Le pipeline MOSS est **vendorisé** dans `third_party/moss_image/` (fichier, LICENSE
  Apache 2.0, commit d'origine, hash SHA-256) : jamais de `trust_remote_code` à
  l'exécution, et une diff lisible si OpenMOSS publie une nouvelle version.
- Avant toute mesure GPU : carte libre (`nvidia-smi`, 0 à 5 %), apps crispz fermées.

## 5. Lots

### Lot 1 : moteur local (cz_moss.py), fichiers uniques, et A/B texte contre Ideogram

1. `MossPipe` autour d'une sous-classe du `MOSSImagePipeline` vendorisé (même convention
   d'appel que `IdeogramPipe`). La sous-classe ajoute : `callback_on_step_end` (tenseur
   `latents`, step joué, total), `_interrupt` (Stop qui rend l'image partielle),
   paramètre `sigmas` (planning tronqué pour le lot 3, en gardant l'alignement sur la
   grille d'entraînement du scheduler d'origine). Rien d'autre n'est modifié dans le
   fichier vendorisé.
2. Chargement depuis `moss_model` (défaut `OpenMOSS-Team/MOSS-Image-SFT`, `Base` au
   choix), sans token. Précision : DiT bf16 (10,2 Go), encodeur `moss_encoder_precision`
   = bf16 ou nf4 (bitsandbytes, à vérifier sur Qwen3.5), VAE fp32. Offload none / model
   (« encodeur seul », repris d'`IdeogramPipe.enable_model_cpu_offload`) / sequential /
   auto. Les hooks de diffusers ne sont pas utilisés (même raison qu'Ideogram : le
   transformer ne doit pas faire d'aller-retour par step).
3. Txt2img : steps (défaut 50, curseur 2 à 80), taille (multiples de 16, 512 à 2048 par
   côté, 4 Mpx max, préréglages à 1 / 1,5 / 2 Mpx), seed, batch (boucle interne du
   transformer : coût linéaire, dit dans l'aide), `guidance_scale` constant (défaut 3,0,
   curseur 0 à 10), **prompt négatif réel** (`supports.negative = True`, preset de tags
   officiel proposé), `uncond_skip_layers` (défaut 22, case « réalisme brut » qui le
   coupe, aide : « plus réaliste, détails qui peuvent s'effondrer »), SLG avancé masqué
   par défaut. `encode_prompt` exposé au cache d'embeddings de `cz_pipeline` (clé :
   prompt + négatif + longueur max).
4. Aperçu live : les tokens sont dans l'ordre FLUX, la matrice FLUX.2 de
   `cz_pipeline` doit s'appliquer sans la correction Ideogram. Test : inverse exact du
   décodage MOSS sur stats `bn` aléatoires, puis mesure sur la carte (erreur sur 0-255
   au dernier step, seuil 40 du script).
5. Garde par famille (`family_from_keys`) : marqueurs forts `video_embedder`,
   `text_embedder.proj`, `shared_block_modulation`, `final_linear_video`,
   `layers.N.proj.q` ; largeur 3072 ; FLUX.2 / FLUX.3 / Ideogram / Z-Image refusés en
   nommant leur studio. LoRAs : lot 4 (pas de mixin dans le pipeline).
6. `cz_protocol` : `negative` True, `img2img` / `inpaint` False jusqu'au lot 3, `edit`
   et `refs` False, `cloud` absent.
6b. **Fichiers uniques (ComfyUI / CivitAI)** : `moss_transformer` (ou le menu Checkpoints)
   accepte un DiT bf16 ou int8 ConvRot ; `text_encoder` accepte un encodeur Qwen3.5 en
   bf16 ou int8 ConvRot, ou un repo / dossier transformers (finetune « abliterated »...).
   Refus nommés pour les 4 bits, les LoRAs posés dans le mauvais dossier et les autres
   familles. Tests CPU sur des fichiers ComfyUI miniatures écrits par les tests eux-mêmes
   (fusion qkv, blobs, quantification int8 ConvRot réelle) relus à l'identique.
7. **`tools/gpu_lot1_check.py`** sur la 5090, carte libre, pour chaque offload utile :
   chargement, VRAM au repos et en pic, 1024x1024, 1456x1456 (2 Mpx), 2048x1152, 50 steps,
   Stop, aperçu live, encodeur bf16 contre nf4 (temps d'encodage et écart). Puis l'**A/B
   texte** : les trois affiches du lot 2 d'Ideogram (`bench/lot2/`, dont « GRAND OPENING »
   et le logo « BEAN THERE ») rendues par MOSS en texte libre et en layout JSON, 4 seeds
   chacune, mêmes tailles ; rapport `bench/lot1/ab.md` : texte exact, texte fantôme en
   fond, banding, temps. Verdict de Mike sur images.

Acceptation : tests CPU sur un MOSS miniature aléatoire (`tests/moss_tiny.py`, vrai
tokenizer Qwen3.5, mêmes tests que `test_ideogram_engine.py` : forme, Stop, aperçu,
négatif, garde), une 2 Mpx sur la 5090, et un A/B jugé « au moins aussi bon qu'Ideogram
sur le texte » par Mike. **Si l'A/B déçoit, on s'arrête là** et crispz-moss reste un
banc d'essai.

### Lot 2 : légendes layout JSON

1. Schéma MOSS dans `cz_caption` : `summary {meta[], canvas, caption}`, `items [{label,
   caption, text?, bbox}]`, `bbox_format: "yxyx"`, `bbox_coord_space: "abs_centralized"`
   (pixels, origine au centre, y vers le bas, x vers la droite ; donc **une légende vaut
   pour une taille** : la taille du canevas est écrite dans la légende et l'UI force
   largeur et hauteur à la suivre). Représentation interne gardée en 0..1000 (celle de
   l'éditeur et du plateau de boîtes) et sérialisation MOSS à la taille choisie ; import
   inverse d'un JSON MOSS collé. Validation par un vérificateur maison (le modèle n'en
   publie pas), `\n` autorisés dans `text`.
2. Presets de design repris (affiche, logo, carte, livre, slide, packaging, miniature,
   story, bannière) aux tailles natives 1 à 2 Mpx, `meta` rempli (`poster`, `logo`...).
3. Mode « tags » (ordre officiel Qualité, Score, Meta, Année, Personnage, Série, Artiste,
   Général) et preset de négatif.
4. Magic prompt, deux moteurs locaux : **Ollama** (`cz_ollama`, modèle `qwen3.5` ou au
   choix) avec le prompt du skill officiel `skills/moss-image-prompt-enhancer` (Apache
   2.0, conçu pour GPT-6, à adapter à un modèle local, schéma passé en `format`) ;
   **encodeur réutilisé** (Qwen3.5-9B déjà chargé + `lm_head` téléchargé seul depuis
   `Qwen/Qwen3.5-9B`, même montage que la tête d'Ideogram, `outlines` pour le JSON) :
   à étudier, défaut Ollama si la tête ne tient pas en VRAM.
5. Describe local : tour vision de Qwen3.5 (présente dans les poids) ; image vers légende
   MOSS avec boîtes ; à vérifier en nf4 et en offload encodeur seul.
6. `tools/gpu_lot2_check.py` : légende à la main, magic prompt, Describe, détecteur de
   texte fantôme (OCR léger si disponible, sinon jugement sur images).

Acceptation : une légende à la main et une légende du magic prompt produisent un JSON
valide et une image dont les textes sont aux boîtes demandées (à l'œil, puis mIoU
approximatif si l'OCR est là).

### Lot 3 : img2img, inpaint, outpaint émulés, et variantes rapides

1. SDEdit sur la fin du planning MOSS (sigmas du scheduler d'origine, alignement sur la
   grille conservé, `strength` = fraction des steps, au moins un), image encodée en tokens
   FLUX (inverse exact du décodage, normalisation `bn`), `scheduler.set_timesteps`
   enveloppé le temps de l'appel. Masque : mélange latent à chaque step, max-pool sur la
   grille de 16 px ; masque plein = img2img simple. Reprise directe de
   `IdeogramPipe._prepare_emulated` avec l'ordre de tokens changé.
2. Débloque refine, tuiles, inpaint, outpaint, reframe, détailleur. Presets de refine et
   d'upscale à vérifier sur la carte (12 à 16 steps à denoise 0,3, hérités ; sur MOSS à
   50 steps nominaux, la règle « 3 ou 4 steps joués suffisent » est à remesurer).
3. **Preset « Draft »** : LoRA 4 steps ou checkpoint DMD 8 steps communautaires
   (`lodestones/MOSS-Image-SFT-8-steps-dmd`), CFG 1, `uncond_skip_layers` coupé ; aide :
   « brouillon, textes de fond à relire ». Dépend du chargeur de LoRA du lot 4 pour la
   LoRA, le checkpoint DMD passe par `moss_transformer`.
4. `FAMILY_CAPS` : img2img et inpaint True, « émulé » dans `caps`.

Acceptation : tests CPU repris de `test_ideogram_emulated.py` ; `tools/gpu_lot3_check.py`
(img2img 0,3 / 0,6, inpaint, outpaint, reframe, Draft) sur la 5090.

### Lot 4 : LoRAs

1. Chargeur de LoRA maison (le pipeline n'a pas de mixin) sur le dialecte de `cz_pipeline`
   : cibles `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`, `text_embedder`, `video_embedder`.
   **Piège relevé** : les poids ComfyUI (Comfy-Org) fusionnent q/k/v et les LoRAs
   communautaires sont souvent à ce format ; un chargeur naïf saute 378 clés en silence.
   Table de correspondance fusionné vers séparé, clés non reconnues **comptées et dites**.
2. Garde par famille (lot 1), refus nommés pour FLUX.2 / FLUX.3 / Ideogram / Z-Image.
3. LoRAs communautaires à essayer, chacune opt-in avec sa licence affichée : DeBand
   (banding du SFT), 4 steps (Draft), couleur / profondeur (Comfy-Org `loras/`), et
   **`chinoll/Moss-Image-Edit`** (édition par instruction avec 1 à 4 images de référence ;
   code GPLv3, licence des poids non dite : étudier d'abord comment les références
   entrent dans la séquence dans ses nœuds ComfyUI, et les blocs modulaires
   `OzzyGT/moss_image_custom_blocks` qui font passer des références par la tour vision).
   Si une voie propre existe : onglet « Edit (references) » réactivé, `supports.edit` et
   `refs` à True. Sinon : lot 5.

Acceptation : un LoRA Ideogram refusé, un LoRA MOSS au format ComfyUI chargé sans clé
perdue (test CPU sur fixtures), DeBand et Draft mesurés sur la carte.

### Lot 5 : outils locaux et passerelles

- Remove BG (rembg) et Vectorize (vtracer) : repris tels quels.
- Handoff : « Edit in crispz-flux3 » repris ; ajout de « Compare in crispz-ideogram »
  (même légende traduite dans le schéma Ideogram, même taille, route CLI du protocole v1)
  pour les A/B à la demande. Ports : crispz-flux3 7860, crispz-ideogram et crispz-moss
  à régler dans les deux configs.

### Lot 6 : tests, doc, livraison

- Suite CPU verte, e2e GPU (`tests/test_moss_e2e.py` : chargement, 1024, 2 Mpx, img2img,
  inpaint, Stop) sur la 5090.
- README : licence Apache 2.0 en tête et **chaîne des licences** (DiT, Qwen3.5-9B, VAE
  FLUX.2 klein 4B, LoRAs tierces) avec ce que ça permet ; tableau des mesures ; A/B
  contre Ideogram ; limites (texte fantôme, banding, RAM de l'offload).
- FORK.md : base crispz-ideogram, différences, mesures.
- Launcher Pinokio `crispz-moss.pinokio` (repo à créer par Mike), icône générée avec
  l'app (preset Logo, enfin commercialisable).

## 6. Licence : à afficher dans l'app et le README

Apache 2.0 sur les poids MOSS-Image et le code du pipeline (fichier LICENSE du repo HF).
Usage commercial permis, attribution et mention des modifications à conserver. Les
composants tiers suivent leur propre licence : Qwen3.5-9B (Apache 2.0), VAE FLUX.2 klein
4B (Apache 2.0, à confirmer sur le repo black-forest-labs), LoRAs communautaires (à lire
une par une ; `Moss-Image-Edit` : code GPLv3, poids non précisés). Pour Supersonique
Studio : livrables clients possibles avec le modèle de base ; les LoRAs tierces sont
vérifiées avant tout usage client.

## 7. Hors périmètre v0.1

- Entraînement de LoRAs (KohakuMossTrainer et DiffSynth font ça très bien à côté).
- Variantes non publiées du rapport (MoE, distillée CFG, vidéo / GIF, référence native).
- SGLang, vLLM-Omni, MLX, GGUF (ComfyUI les couvre).
- Tout cloud : il n'en existe pas pour ce modèle, et on n'en veut pas.

## 8. Risques connus avant de commencer

- Texte : résultats tiers contradictoires ; l'A/B du lot 1 décide.
- RAM : l'offload « encodeur seul » garde 16,8 Go d'encodeur en RAM ; sous 48 Go de RAM
  système, préférer l'encodeur nf4 (à valider) ou l'offload sequential.
- Qwen3.5 sous Windows : couches GDN sans noyaux Triton, repli torch ; l'encodage d'un
  prompt long (4096 tokens max) est à chronométrer.
- Banding du SFT : montrer la case DeBand dès le lot 4 ; Base en secours.
- Un `config.txt` repris de crispz-ideogram garde 48 steps / guidance 7,0 : à écraser au
  premier lancement (MOSS : 50 / 3,0), avec un message.

## 9. État au 03/11/2026

- Repo créé par Mike, `main` = `main` de crispz-ideogram (4d028a3, 0.1.0) avec ses 446
  commits, plus ce cahier des charges. Le `CLAUDE.md` d'Ideogram est gardé en
  `FORK-ideogram-CLAUDE.md` (pièges d'environnement, mesures, décisions : à lire avant
  tout test GPU), avec `FORK-flux3-CLAUDE.md` en dessous.
- Remotes : `origin` = crispz-moss, `ideogram` = crispz-ideogram.
- Cahier validé par Mike le 03/11/2026, avec l'ajout de la compatibilité Hugging Face et
  CivitAI (§2, fichiers uniques).

### Lot 1 (03/11/2026) : moteur livré sur `wip/moss-engine`, mesure GPU à faire

- `third_party/moss_image/` : `pipeline_moss_image.py` d'OpenMOSS vendorisé tel quel
  (Apache 2.0, LICENSE, README avec commit et SHA-256), jamais de `trust_remote_code`.
- `cz_moss.py` : `MossPipeline` (sous-classe, la boucle amont ligne pour ligne plus
  `callback_on_step_end`, `_interrupt`, `sigmas` ; test : même image au pixel près que la
  classe amont), `MossPipe` (convention d'appel de crispz, négatif réel avec cache du
  dernier négatif, `encode_prompt` pour le cache d'embeddings, offload « encodeur seul »
  repris d'Ideogram, refus nommés img2img / inpaint (lot 3), édition (crispz-flux3) et
  LoRA (lot 4), couches `uncond_skip` ramenées à celles du modèle), détection de famille
  (`video_embedder`, `.proj.q.` ... ; Ideogram et FLUX nommés via cz_ideogram), `load`
  composant par composant (DiT et scheduler vendorisés, `Qwen3_5Model`, VAE fp32),
  lecteurs de fichiers uniques ComfyUI / CivitAI (DiT bf16 / int8 ConvRot, encodeur
  bf16 / int8 ConvRot via `from_pretrained(None, state_dict=...)`, refus 4 bits par nom).
- `cz_pipeline` : `engine = cz_moss`, clés `moss_model`, `moss_transformer`,
  `moss_precision` (= précision de l'ENCODEUR, bf16 ou nf4 ; le DiT reste bf16), env
  `MOSS_*`, guidance 3,0, négatif passé, aperçu live sur les latents (B, 128, h, w) avec
  la matrice FLUX.2 telle quelle (ordre FLUX, test), VRAM estimée 25,4 Go (bf16) ou
  15,4 (nf4), 'auto' -> 'model' encodeur seul sous 40 Go, encodeur de remplacement
  (`text_encoder`) appliqué, fichiers uniques listés et chargés, LoRA MOSS refusée avec
  « lot 4 », Ideogram / FLUX refusés en nommant leur studio.
- `cz_protocol` : `crispz-moss`, `negative` True, img2img / inpaint False, cloud toujours
  off. `cz_core` : profil `moss` 50 / 3,0. `config-sample.txt`, `cz_ui`, `cz_cli`,
  `preferences.example.json` : champ négatif rétabli, presets « MOSS (50 steps) » et
  « Draft (25 steps) », notes de licence Apache 2.0 par repo (SFT, Base), sélecteur de
  précision de l'encodeur visible, filtre CivitAI « MOSS Image », `--moss-model` /
  `--moss-transformer` (anciens noms en alias).
- Tests : `tests/moss_tiny.py` (MOSS miniature : DiT 2 blocs, Qwen3.5 4 couches hybride
  avec les replis torch, VAE FLUX.2 miniature, vrai tokenizer), `test_moss_engine.py` (20),
  `test_moss_wiring.py` (12) ; smoke 22/22.
- **Tests hérités d'Ideogram, traités un par un** (aucun désactivé, aucun xfail) :
  - réécrits pour MOSS, même intention : `test_civitai_search` (base « MOSS Image »,
    case « MOSS only »), `test_detailer_steps` (profil 50), `test_encoder_trim` (budget
    bf16 ~25 Go), `test_flux3_wiring`, `test_klein_variant` (fixture MOSS acceptée,
    Ideogram refusé en nommant crispz-ideogram, LoRA MOSS refusée « lot 4 »),
    `test_preset_model_switch` (deux repos, notes Apache 2.0, largeur 3072),
    `test_quant_formats` (clé `video_embedder`, GGUF MOSS refusé par nom),
    `test_model_profiles` (clés `moss_*`, 50 / 3,0, précision bf16 / nf4),
    `test_protocol`, `test_variants_wiring` et `test_improve_wiring` (négatif réel),
    `test_undistilled_guidance`, `test_ideogram_api` (le protocole ne signale jamais de
    cloud) ;
  - fixtures de LoRA d'une famille inconnue (jamais filtrée) pour les tests de mécanique
    générique `test_lora_dialect`, `test_lora_offload` : une LoRA MOSS est refusée par
    nom jusqu'au lot 4, elle n'atteindrait pas ce qui est testé ; `test_prescaled_fp8`
    (marqueur `video_embedder`) ;
  - **retiré** : `test_ideogram_wiring.py` (câblage de l'IdeogramPipe dans cz_pipeline,
    remplacé par `test_moss_wiring.py` ; ses tests des chemins img2img / refine /
    outpaint reviennent au lot 3 sur MossPipe). `test_ideogram_engine.py` et
    `test_ideogram_emulated.py` restent (ils testent l'adaptateur Ideogram seul, encore
    dans l'arbre) jusqu'au retrait de `cz_ideogram.py` (lot 2 ou 6).
- Suite complète (`tools/run_tests.py`, un processus par fichier, ce que lance le CI) :
  **69 fichiers sur 69 passent** (dont `test_ideogram_e2e` sauté sans CUDA) ; smoke 22/22 ;
  `check_config_keys`, `check_lock`, `compileall` OK. En un seul processus (`pytest tests`),
  deux tests tombent sur l'état global partagé entre fichiers (la raison d'être de
  `run_tests.py`, déjà vraie avant le fork).
- Vérifié dans Chromium : l'app démarre, champ négatif actif avec la suggestion MOSS,
  bouton « Improve negative », titre `crispz-moss 0.1.0-dev`. Reste visible, pour le
  lot 2 : l'accordéon « Ideogram caption (JSON) » (l'éditeur garde le schéma Ideogram
  jusqu'au lot 2).
- `tools/gpu_lot1_check.py` réécrit : chargement, 1024 à froid et à chaud, 1456x1456,
  2048x1152, aperçu, Stop, `--nf4` (encodeur), puis l'**A/B texte** (3 designs du lot 2
  d'Ideogram en texte libre et en layout JSON, 4 seeds, `bench/lot1/ab.md` à remplir à
  l'œil, liens vers `../crispz-ideogram/bench/lot2`).
- Sous Windows, transformers avertit « causal_conv1d / flash-linear-attention absents,
  repli PyTorch » à chaque encodage : exact, plus lent, à chronométrer (`2_encode_prompt_s`
  du rapport).
- À faire chez Mike : `git pull`, `pip install` rien de nouveau (bitsandbytes seulement
  pour `--nf4`), app crispz-ideogram FERMÉE, `tools\gpu_lot1_check.py` ; verdict sur
  `bench/lot1/ab.md`.

### Premier passage sur la 5090 (03/11/2026, Mike) : mesures du lot 1

`tools/gpu_lot1_check.py`, offload 'auto' -> 'model' encodeur seul, encodeur bf16, 50 steps,
CFG 3,0, torch 2.11 cu128, premier lancement (téléchargement des 27 Go compris dans le
chargement) :

| | valeur |
|---|---|
| Chargement (téléchargement compris) | 278,7 s ; au repos 9,7 Go alloués (DiT + VAE), 13,1 Go vus par le pilote |
| 1024x1024 à froid | 59,8 s (1,20 s/step), pic 26,95 Go réservés (l'encodeur de 16,8 Go est sur la carte pendant l'encodage, puis repart) |
| 1024x1024 à chaud | **42,7 s (0,855 s/step)**, pic 15,3 Go (prompt en cache d'embeddings : l'encodeur ne revient pas) |
| 1456x1456 (2 Mpx) | 101,7 s (2,03 s/step), pic 15,6 Go |
| 2048x1152 | 115,7 s (2,32 s/step), pic 15,7 Go |
| Encodage d'un prompt | 5,2 s (aller-retour PCIe de l'encodeur compris, noyaux GDN en repli torch) |
| Décodage VAE | 0,2 à 0,6 s |
| Stop | image partielle 0,8 s après la demande (step 19 sur 50) |
| Aperçu live, dernier step | erreur 34,6 sur 255 (matrice FLUX.2 telle quelle), verdict OK |

Contre Ideogram 4 sur la même carte : 1024 en 42,7 s contre 61 s, 2 Mpx en 102 s contre
394 s pour 4 Mpx (2048x2048 non mesuré ici), pic 15,7 Go contre 23,4. Lecture :
- Le pic à 27 Go n'arrive que quand l'encodeur est sur la carte (prompt ou négatif
  nouveau) ; en régime de prompts variés il revient à chaque image, et il reste 5 Go de
  marge sur 31,8 : un navigateur gourmand suffit à faire déborder. Deux pistes, à
  mesurer : l'encodeur en nf4 (`--nf4`, bitsandbytes présent) qui ramène ce pic vers
  16 Go, ou encoder prompt et négatif AVANT de monter le DiT (impossible : le DiT est
  résident par construction de l'offload 'model' ; l'offload 'sequential' le ferait).
- Noyaux SDPA à head_dim 128 : `efficient` et `cudnn` oui, `flash` NON sur cette build
  (torch 2.11 cu128, Blackwell) ; rien à faire côté app.
- L'encodeur Qwen3.5 tourne avec les replis torch (`flash-linear-attention`,
  `causal_conv1d` absents) : 5,2 s par prompt, dont le transfert de 16,8 Go. Acceptable ;
  à comparer au même prompt en nf4.
- A/B texte : 24 rendus produits dans `bench/lot1/ab/`, 25 s par affiche 720x1024 et
  39 s par logo 1024 (0,5 à 0,8 s/step), texte libre et layout JSON à égalité de temps.

### Bilan du lot 1 (03/11/2026) : A/B texte gagné, lot VALIDÉ

Lecture sur images des 24 rendus de l'A/B (envoyés par Mike, lus en session) :

| design | rendus | textes demandés exacts | texte ajouté non demandé | banding |
|---|---|---|---|---|
| logo « BEAN THERE » 1024 | 8 | 8 / 8 | 3 / 8, en texte libre seulement : « COFFEE SHOP » déduit du prompt, dont 2 fautes (« COFE », « COFFE ») | aucun |
| affiche boulangerie 720x1024 (« GRAND OPENING », « SATURDAY 9 AM · RUE DU PAIN ») | 8 | 8 / 8, point médian et français compris | aucun | aucun |
| affiche suisse 720x1024 (« 5B », « MOSS, real typography ») | 8 | 8 / 8, virgule comprise (une fois sur deux lignes) | aucun | aucun |

24 rendus sur 24 avec tous les textes demandés exacts, y compris les petits textes fixes
que le test ai-muninn donnait comme point faible. Composition tenue (titre, ruban,
panier ; numéraux, cercle rouge qui chevauche, légende en bas à gauche), styles variés
d'un seed à l'autre sans casser le texte, aucun banding sur le SFT à ces tailles.
Ideogram 4 avait réussi la même affiche au lot 2 en 56 s ; MOSS la rend en 25 s.
**Verdict : au moins aussi bon qu'Ideogram 4 sur le texte demandé, et plus rapide.**

Le seul défaut observé, et il est instructif : en TEXTE LIBRE, quand le prompt dit
« coffee shop logo », le modèle ajoute parfois « COFFEE SHOP » dans l'image sans qu'on
l'ait demandé, et l'orthographie mal deux fois sur trois (texte fantôme « local », pas
du charabia en fond). En LAYOUT JSON, où les textes sont listés un par un, aucun ajout
sur 4 rendus. Conséquences pour le lot 2 : le layout n'est pas qu'un outil de placement,
c'est la garantie « rien d'autre que ces mots » ; le preset Logo doit passer par lui ;
et le mode texte libre peut glisser « no other text » quand la légende le permet
(à mesurer).

Critères d'acceptation du lot 1 remplis : tests CPU sur MOSS miniature, une 2 Mpx sur la
5090 (1456x1456 en 102 s), A/B jugé. Points ouverts, aucun bloquant : encodeur nf4 non
mesuré (pic à 27 Go sur un prompt nouveau), temps de chargement depuis le cache non noté
(le premier passage incluait le téléchargement).

### Deuxième passage sur la 5090 (03/11/2026) : cache chaud et encodeur nf4

`--nf4 --skip-ab --skip-2mpx`, même machine, modèle déjà en cache :

| | bf16 (encodeur en RAM, offload 'model') | nf4 (encodeur bitsandbytes, reste sur la carte) |
|---|---|---|
| Chargement depuis le cache | 28,5 s (26,2 s au rechargement) | 29,6 s |
| VRAM au repos (torch) | 9,7 Go | 15,5 Go (DiT + VAE + encodeur nf4 ~5,8 Go) |
| Encodage d'un prompt | 3,9 s (aller-retour PCIe compris) | **0,27 s** (aucun transfert) |
| 1024x1024 à chaud | 42,7 s (0,854 s/step) | 41,5 s (0,83 s/step) |
| Pic réservé, débruitage / décodage | 10,4 / 15,3 Go (27 Go au premier rendu, encodeur sur la carte) | 16,2 / 21,0 Go, **constant** (plus de pic à 27 Go) |
| Écart de pixels contre le rendu bf16 (même seed) | | moyenne 29,6 sur 255, max 221 |

Lecture : en nf4 l'encodeur reste résident (bitsandbytes ne déplace pas ses modules,
`cz_moss._is_quantized`), ce qui supprime le pic à 27 Go et l'aller-retour de 16,8 Go
à chaque nouveau prompt : plafond stable à 21 Go, encodage quasi gratuit. Le prix : des
embeddings différents, donc une image nettement différente au même seed (écart moyen
29,6 : une autre composition, pas un bruit). Décision en attente du jugement sur images
(`bench/lot1/gen1024_nf4_encoder.png` contre `gen1024_warm.png` : textes exacts ?) ;
tant qu'il n'est pas rendu, **bf16 reste le défaut**, nf4 est une option mesurée.
Le décodage VAE en fp32 coûte environ 5 Go de pic à 1024 dans les deux modes.

### Lot 2 (03/11/2026) : layouts JSON, livré sur `wip/moss-engine`, mesure GPU à faire

- **Trois formes du même layout** dans `cz_caption` (réécrit, plus aucun import du
  vérificateur Ideogram) : ÉDITEUR (boîtes en 0..1000, ce que manipulent le formulaire et
  le plateau : un preset ne dépend pas de la taille), DRAFT (la forme lisible du skill
  officiel : boîtes en pixels `xyxy` depuis le coin haut gauche, `width` / `height`
  écrits dedans ; c'est ce que « Use as prompt » met dans la boîte de prompt), COMPILÉ
  (ce que lit le modèle : boîtes centrées `yxyx`, une ligne, sans taille ; même
  arithmétique que `compile_prompt.py` du skill, vérifiée sur ses nombres publiés).
  `prepare_prompt` dans `cz_moss` compile le draft au moment du rendu et **force la
  taille du draft** (une légende vaut pour une taille, dit une fois) ; appelé dans
  `cz_pipeline._qwen_call` AVANT le cache d'embeddings. Un layout compilé collé à la
  main passe tel quel (note si ses boîtes dépassent la taille demandée).
  Validation maison : les contrôles du compilateur officiel + boîte sous 16 px, texte
  vide. Précision : l'éditeur garde des millièmes, un pixel de jeu au retour.
- `cz_design` : les 9 presets réécrits (meta, caption résumé avec « no other text »,
  items), tailles 0,6 à 1,9 Mpx (affiche 1152x1632, logo 1024, carte 1536x992, livre
  1088x1632, slide 1600x896, packaging 1216, miniature 1280x720, story 1024x1824,
  bannière 1920x320). Le preset Logo passe par le layout (leçon de l'A/B).
- `cz_caption_ui` : même plateau et même pont JS, formulaire résumé / meta / canvas /
  palette / items (label, text, caption, bbox, palette), JSON brut = le draft (un compilé
  est accepté, la taille des curseurs sert de canevas), « Use as prompt » et « Apply »
  règlent largeur et hauteur ; Describe règle la taille sur celle de l'image.
- `cz_magic` : deux moteurs locaux. **Ollama** (défaut) : brief adapté du skill officiel
  (`SYSTEM`), schéma du draft en `format`, URL loggée à chaque appel. **Local** : le
  `Qwen3_5Model` déjà chargé devient génératif via `Qwen3_5ForConditionalGeneration`
  construit sur meta avec l'encodeur partagé (aucune copie) et le `lm_head` de
  `Qwen/Qwen3.5-9B` : **un seul tenseur** (248320 x 4096, ~2 Go bf16) téléchargé par
  requête HTTP Range sur le shard 1 (repli : le shard entier), rangé dans
  `cache/moss/qwen3.5-9b-lm_head.safetensors`. JSON contraint par outlines 1.3
  (`outlines.from_transformers` + `get_json_schema_logits_processor`, API changée depuis
  Ideogram), dit une fois sinon ; dynamo coupé sans Triton. Nettoyage commun : think,
  fences, réponse dessinée pour une autre taille (remise à l'échelle, dit), réponse en
  millièmes, réponse déjà compilée. Describe local : processeur image de
  `Qwen/Qwen3.5-9B`, boîtes en pixels de l'image puis draft sur la grille 16.
  **Non mesuré sur la carte** : l'encodeur nf4 avec le lm_head bf16, la vitesse de
  génération avec les replis torch (le magic prompt Ideogram prenait 10 à 20 s).
- **Cloud Ideogram retiré** : `cz_ideogram_api.py`, `cz_cloud_ui.py`,
  `tests/test_ideogram_api.py`, `tools/api_lot4_check.py`, `third_party/ideogram4`, les
  clés `ideogram_cloud_*` / `ideogram_api_key`, l'onglet et le bouton Describe cloud.
  `supports.cloud` reste dans `caps` (contrat de famille), toujours off.
- Tests : `test_caption.py` (12 : nombres du skill et du guide, aller-retour, refus,
  presets, `prepare_prompt`, éditeur), `test_magic_prompt.py` (11 : nettoyage, Ollama
  mocké, plomberie réelle du magic prompt et du Describe sur le MOSS miniature avec une
  tête aléatoire, téléchargement Range contre un faux serveur, cache, mauvaise taille
  refusée, outlines), `test_lot2_tools.py` (2). `tools/gpu_lot2_check.py` : layouts à la
  main (logo + affiche, boîtes dessinées sur les rendus), magic prompt (`--engine`),
  Describe, `--no-other-text` (8 rendus texte libre avec / sans « No other text. »).
- Suite complète (`tools/run_tests.py`) : **68 fichiers sur 68 passent** ; smoke 22/22 ;
  `check_config_keys` (122 clés), `compileall` OK.
- Vérifié dans Chromium : preset Logo chargé, deux boîtes sur le plateau, « Use as
  prompt » écrit le draft et sa taille, statut « Layout: valid ».
- À faire chez Mike : `git pull`, `tools\gpu_lot2_check.py --engine local` (télécharge
  le lm_head, 2 Go) puis `--engine ollama` si un modèle Ollama est là ; juger les
  `*_boxes.png` (textes dans les boîtes ?) et les drafts du magic prompt.

### Premier passage GPU du lot 2 (04/11/2026, Mike) : layouts et magic prompt OK, Describe corrigé

`tools/gpu_lot2_check.py --engine local`, offload 'model' encodeur seul, encodeur bf16,
50 steps, CFG 3,0, outlines présent, 9 minutes en tout :

| étape | temps | résultat |
|---|---|---|
| layout à la main, logo 1024 | 84,4 s | « BEAN THERE » exact, texte et symbole dans leurs boîtes |
| layout à la main, affiche 1152x1632 | 89,7 s | « GRAND OPENING », « SATURDAY 9 AM · RUE DU PAIN » exacts, les trois boîtes tenues |
| magic prompt local, affiche suisse | 113,9 s (dont lm_head 2 Go et compilation du schéma) + rendu 30,8 s | JSON valide, « 5B » et « MOSS, real typography » présents |
| magic prompt local, logo café | 76,9 s + rendu 43,5 s | JSON valide, « BEAN THERE » seul, aucun « COFFEE SHOP » ajouté |
| Describe local du logo | échec | JSON invalide (ligne 21, virgule attendue) |

Lecture sur images (les six rendus, lus en session) :
- Layouts à la main : 3 textes sur 3 exacts, chaque texte et chaque objet dans sa boîte
  (`*_boxes.png`), composition propre. **Critère d'acceptation du lot 2 atteint** sur la
  partie « légende à la main ».
- Magic prompt : 2 JSON valides sur 2, textes demandés présents, aucun texte ajouté ; le
  layout a bien joué son rôle de garde-fou sur le logo café (leçon du lot 1 confirmée).
  Placement : sur l'affiche suisse, Qwen a dessiné la boîte « 5B » trop étroite (le 5
  seul) et le modèle a gardé le texte entier en débordant ; « MOSS, real typography »
  est rendu 25 px au-dessus de sa boîte. Sur le logo, texte et tasse dans leurs boîtes.
  Le modèle tient le texte avant la boîte : à dire dans l'aide de l'éditeur.
- Le logo à la main (84 s pour 1024, contre 39 s en A/B) est le premier rendu du
  processus : prompt et négatif nouveaux (deux allers-retours de l'encodeur de 16,8 Go),
  noyaux CUDA à chaud. Les rendus suivants retrouvent les temps du lot 1 (+4 s
  d'encodage par prompt nouveau).
- **Describe : cause trouvée et corrigée.** La description tournait SANS contrainte de
  schéma (seul le magic prompt passait le guide outlines). Désormais `describe_image`
  est contraint par le même schéma (`constrained=True`), `cz_magic.clean` répare une
  réponse coupée ou malformée (virgules traînantes, chaîne, clé, tableaux et objets
  laissés ouverts ; dit dans les notes « JSON repaired », derniers items à relire), la
  réponse brute entière va dans le log et, pour le script, dans
  `bench/lot2/describe_raw.txt`. Tests : +2 (`test_describe_is_constrained_to_the_draft_schema`,
  `test_clean_repairs_a_cut_or_malformed_answer`).
- Magic prompt local : 77 s par brief, c'est la génération token par token de Qwen3.5
  avec les replis torch (`causal_conv1d_update`, `fused_recurrent_gated_delta_rule`).
  Les noyaux manquants ne s'installent pas sous Windows sans compiler (causal_conv1d :
  nvcc + MSVC ; flash-linear-attention : Triton à la version exacte de torch 2.11) :
  refusé, trop risqué pour le venv. Réponse : le moteur **Ollama**, déjà par défaut,
  à mesurer (`--engine ollama`). Le moteur local reste utile sans Ollama.
- **Moteur Ollama mesuré** (`--engine ollama`, lancé juste après, AVANT le correctif du
  Describe) : affiche suisse 110,6 s (premier appel : Ollama charge son modèle sur une
  carte où le DiT occupe déjà 10 Go), logo café **12,4 s** ; 2 JSON valides sur 2, textes
  présents, rendus 31,8 et 44,2 s. Soit 6 fois plus vite que le moteur local (77 s) en
  régime établi : **Ollama reste le défaut**, le moteur local est le secours sans Ollama.
  Le rapport n'écrivait pas le modèle Ollama utilisé : ajouté (`0_env.ollama_model`).
  Layouts à la main identiques (89,0 s et 91,9 s : le premier rendu à froid coûte
  toujours 85 à 90 s).
- **Troisième passage, après le correctif** (`--engine ollama`, 17:57, 6 minutes) :
  Describe local du logo **78,3 s, JSON valide, aucune note, « BEAN THERE » relu,
  2 items** (symbole et texte). Magic prompt Ollama 20,1 s et 13,4 s (modèle déjà
  chargé), JSON valides, textes présents ; layouts à la main 75,2 s et 90,0 s.
  Le Describe à 78 s, c'est la même génération token par token que le magic prompt
  local (replis torch) ; acceptable pour un bouton ponctuel, et seule voie pour lire
  l'image sans second modèle en VRAM. Piste si Mike le juge lent : Describe par Ollama
  avec un modèle à vision (même brief, image en base64), à décider plus tard.

### Bilan du lot 2 (04/11/2026) : critères d'acceptation atteints, verdict de Mike attendu

Acceptation du cahier : « une légende à la main et une légende du magic prompt
produisent un JSON valide et une image dont les textes sont aux boîtes demandées ».
- Légende à la main : 2 rendus, 3 textes sur 3 exacts, chaque texte et chaque objet
  dans sa boîte (vu sur `hand_logo_boxes.png`, `hand_poster_boxes.png`).
- Magic prompt : 4 drafts valides sur 4 (2 local, 2 Ollama), textes demandés présents,
  aucun texte ajouté ; placement tenu sur le logo, approximatif sur l'affiche suisse
  (boîte « 5B » dessinée trop étroite par le LLM, texte entier gardé par MOSS).
- Describe local : valide, textes relus, boîtes à juger sur `describe_logo_boxes.png`.
- Moteurs : Ollama par défaut (12 à 20 s), local en secours (77 s), Describe local 78 s.
Reste à juger par Mike : `describe_logo_boxes.png` (boîtes sur le symbole et le texte ?)
et les drafts `magic_*.json` à l'œil. Puis feu vert pour le lot 3.

## 10. Sources

- https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT (pipeline, configs, LICENSE)
- https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- https://github.com/OpenMOSS/MOSS-Image (README, USERGUIDE.md, skill du prompt enhancer)
- https://arxiv.org/html/2610.23374 (rapport technique)
- https://huggingface.co/Comfy-Org/MOSS-Image (repack, int8, LoRAs tierces)
- https://huggingface.co/chinoll/Moss-Image-Edit
- https://huggingface.co/OzzyGT/moss_image_custom_blocks
- https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd
- https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand
- https://ai-muninn.com/en/blog/moss-image-traditional-chinese-comic-cards (5090, int8)
- https://www.mindstudio.ai/blog/moss-image-5b-tested-realism-text
