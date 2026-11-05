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
| Licence | Apache 2.0, poids et code de pipeline ; encodeur Qwen3.5-9B Apache 2.0 ; VAE repris de FLUX.2 klein 4B (Apache 2.0 au niveau du repo, confirmé au lot 6) |
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
4B (Apache 2.0 au niveau du repo, confirmé le 05/11/2026), LoRAs communautaires (à lire
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
- Modèle Ollama de Mike (quatrième passage, 18:04, mêmes résultats : Describe 76,1 s
  valide, magic 20,4 et 12,8 s) :
  `tinyrick/gemma-4-31B-it-uncensored-heretic-vision-llmfan46:Q4_K_M`, un Gemma 4 31B
  Q4_K_M (~18 Go) À VISION, qui cohabite sur la carte avec le DiT résident (10 Go) : le
  premier appel de 110 s était son chargement. Conséquence : le « Describe par Ollama »
  (image en base64, même brief) devient immédiat avec ce modèle, sans rien ajouter ;
  à mesurer si Mike le veut (le Describe local reste la voie sans Ollama). Note : un
  finetune communautaire de Gemma suit les conditions Gemma, qui ne revendiquent rien
  sur les sorties ; il ne produit que du texte de brief, jamais les images.
**Lot 2 VALIDÉ par Mike le 04/11/2026**, avec une rallonge demandée : le Describe par
Ollama.
- **Describe par Ollama** (`cz_magic.describe(image, which)`, rendu (texte, notes,
  moteur)) : suit le sélecteur de moteur du magic prompt. En mode Ollama, le modèle du
  magic prompt est pris s'il a la capacité `vision` selon `/api/show`, sinon le premier
  modèle vision installé (`cz_ollama._ollama_vision_models`) ; sans modèle vision, repli
  sur le local, dit dans le log. Image en JPEG 1024 px base64, même brief que le local
  (taille d'origine écrite dedans), schéma du draft en `format`, température 0. L'UI dit
  le moteur utilisé ; `gpu_lot2_check.py --engine ollama` décrit par Ollama et l'écrit
  dans le rapport. Test : `test_describe_through_an_ollama_vision_model`. À mesurer chez
  Mike avec son Gemma 4 31B vision (attendu : 15 à 25 s contre 76 s en local).

### Lot 3 (04/11/2026) : img2img, inpaint, outpaint émulés, livré sur `wip/moss-engine`, mesure GPU à faire

- `cz_moss.MossPipe._prepare_emulated` : SDEdit sur la fin du planning propre de MOSS
  (`native_sigmas` : sigmas décalées puis alignées sur la grille d'entraînement, doublons
  retirés, calculées sur le scheduler du pipe), `strength` = fraction des steps RÉELS
  (après dédoublonnage ; au moins un step), image encodée en latents MOSS par l'inverse
  exact de `decode_latents` (`pack_latents` : VAE FLUX.2, paquets 2x2 dans l'ordre FLUX
  (c, p, p), normalisation `bn`), bruitée au sigma du premier step gardé, planning
  tronqué passé en `sigmas` (sans décalage supplémentaire : la méthode de lot 1). Pas de
  rotation de guidance (constante sur MOSS). Masque : mélange latent à chaque step
  (`known = (1 - sigma_suivant) * x0 + sigma_suivant * bruit` hors masque), max-pool sur
  la grille de 16 px, masque tout blanc = img2img simple. La barre de progression
  parcourt les timesteps tronqués (juste d'elle-même), l'aperçu live lit
  `_cz_run_steps`, remis à None après l'appel ; le scheduler n'est pas enveloppé (le
  paramètre `sigmas` du lot 1 suffit). Une image SANS strength ni masque reste un refus
  nommé (édition : crispz-flux3).
- **Un layout en img2img suit l'image** : `prepare_prompt(..., fit=True)` redessine un
  draft fait pour une autre taille aux dimensions de l'image (boîtes remises à l'échelle,
  dit dans le log) au lieu de forcer sa taille ; `cz_pipeline._qwen_call` passe `fit`
  dès qu'un `strength` ou un `mask_image` est là (et prend la taille de l'image si
  l'appelant ne la donne pas). Un rendu texte vers image garde la règle « un layout vaut
  pour une taille ».
- `cz_protocol` : `FAMILY_CAPS` img2img et inpaint à True, `supports.emulated` =
  `["img2img", "inpaint"]`. Débloque refine, upscale par tuiles, inpaint, outpaint,
  reframe et détailleur de l'app sans autre changement (chemins `inpaint_run`,
  `_refine_whole`, `reframe`, `outpaint_directions` testés sur le MossPipe miniature).
- **Preset « Draft » : décision reportée au lot 4, avec la raison.** Le checkpoint DMD
  8 steps (`lodestones/MOSS-Image-SFT-8-steps-dmd`, lu le 04/11/2026) n'est pas un
  transformer à charger à 8 steps : sa voie officielle est une LoRA kohya (rang 128 sur
  `proj.{q,k,v,o,ffn_up_gate,ffn_down}`) PLUS des remplacements fp32 de `time_embedding`
  et des biais de modulation, PLUS un échantillonneur maison (mélange latent par une
  matrice Q 128x128 et des codes de step, pas le scheduler du modèle) ; son dossier
  `transformer/` est marqué obsolète par l'auteur et exige le fp32. Donc : chargeur de
  LoRA du lot 4, puis un sampler DMD dédié si le gain le justifie. En attendant,
  `gpu_lot3_check.py` rend le preset affiche à 12, 20, 25 et 50 steps (même seed) pour
  savoir quel « Draft » du modèle de base vaut la peine ; l'aide de `config-sample.txt`
  le dit.
- Tests : `tests/test_moss_emulated.py` (11 : fin du planning natif et retour au planning
  entier à l'appel suivant, extrêmes de strength, image plus proche à faible strength,
  inverse exact du décodage sur tokens aléatoires, latents hors masque EXACTEMENT gardés,
  masque plein, Stop qui rend l'image partielle, max-pool, refus de l'édition, compteurs,
  négatif + batch de 2 en img2img) ; `test_moss_wiring` +4 (chemins de l'app, draft
  remis à l'échelle en img2img, caps) ; `test_moss_engine` : le refus « lot 3 » devient
  le test de l'édition seule. Suite complète : **69 fichiers sur 69** ; smoke 22/22 ;
  `check_config_keys` OK.
- `tools/gpu_lot3_check.py` réécrit pour MOSS : init = `bench/lot2/hand_poster.png` et
  son draft ; img2img 0,3 / 0,6 (même layout, remis à l'échelle), inpaint (vélo rouge
  dans une boîte centrale), outpaint à droite et reframe 16:9 (légende du preset en texte
  libre), A/B refine 0,3 à 12 et 36 steps (la question du lot 7 d'Ideogram, à remesurer
  à 50 steps nominaux), rendus Draft à 12 / 20 / 25 / 50 steps (`--skip-draft`). Temps et
  pics dans `bench/lot3/report.md`.
- À faire chez Mike : `git pull`, carte libre, `.venv\Scripts\python tools\gpu_lot3_check.py` ;
  juger sur images : texte gardé en img2img 0,3 et 0,6, zone hors masque intacte, raccords
  de l'outpaint et du reframe, 12 contre 36 steps en refine, et à partir de combien de
  steps le texte du Draft tient.

### Premier passage GPU du lot 3 (04/11/2026, Mike) : émulation validée sur images, deux défauts vus

`tools/gpu_lot3_check.py`, init = l'affiche du lot 2 (1152x1632, 1,9 Mpx) et son draft,
offload 'model' encodeur seul, encodeur bf16, 50 steps nominaux, 8 minutes en tout :

| étape | temps | pic réservé | taille | lecture sur image |
|---|---|---|---|---|
| img2img 0,3 (15 steps) | 71,0 s (premier rendu du processus : encodeur pour le prompt ET le négatif, noyaux à froid) | 30,0 Go | 1152x1632 | affiche intacte, textes exacts |
| img2img 0,6 (30 steps) | 52,3 s | 30,0 Go (réservé non rendu depuis le rendu précédent) | 1152x1632 | une seule image reçue pour les deux : écart 0,3 / 0,6 non jugé |
| inpaint (boîte centrale, « red retro bicycle ») | 43,4 s | 30,0 Go (idem) | 1152x1632 | vélo rouge net dans le panier, hors masque intact au pixel, raccords propres, textes intacts |
| outpaint à droite | 43,5 s | 27,2 Go | 1497x1632 | **zone neuve remplie de faux textes** (« EAR. », « BIGNE EIFFOR VEWAY », « BENLO REMORIES FOOM ») et d'une plante |
| reframe 16:9 (contain) | 39,7 s | 14,3 Go | 1376x768 | **bandes latérales remplies de gros faux titres** (« WIOIJAVE », « HRNJOSY! ») |
| refine 0,3 à 12 steps | 7,8 s | 15,8 Go | 1152x1632 | images non reçues |
| refine 0,3 à 36 steps | 19,6 s | 15,8 Go | 1152x1632 | images non reçues |
| Draft 12 / 20 / 25 / 50 steps (preset affiche, textes de gabarit) | 24,8 / 34,4 / 42,9 / 85,0 s | 25,4 (prompt neuf) puis 15,5 Go | 1152x1632 | **« HEADLINE » et « DATE · PLACE » exacts aux quatre** ; à 12 steps le visage et le col diffèrent un peu, rien de cassé |

Lecture :
- **L'émulation marche** : img2img garde l'affiche et ses textes, l'inpaint pose le vélo
  dans la boîte sans toucher au reste, aucune couture. Critère d'acceptation GPU du lot 3
  atteint sur img2img et inpaint. Inpaint, outpaint et reframe travaillent sur une zone
  plafonnée à ~1 Mpx (`_cap_work_res`), d'où des temps sous ceux d'un rendu 1,9 Mpx.
- **Outpaint et reframe inventent du texte** dans les zones neuves. Le prompt était la
  légende du preset en texte libre (avec « no other text ») : la leçon du lot 1 se
  répète, seul le LAYOUT tient « rien d'autre que ces mots ». Et quand un draft est
  passé en img2img, `fit=True` l'ÉTIRE sur le nouveau canevas : les boîtes de texte
  ne tombent plus sur l'image d'origine. Correctif à faire (lot 3b) : placer le draft
  au lieu de l'étirer, c'est-à-dire reporter ses boîtes dans le sous-rectangle où
  l'image d'origine est collée (outpaint : décalage des marges ; reframe : `ox, oy` du
  contain), à l'échelle de la zone de travail, et garder « no other text » ; le mode
  texte libre ajoute « No other text. » si la légende ne le dit pas. À mesurer sur les
  mêmes rendus.
- **Pic à 30 Go** sur les premiers img2img à 1,9 Mpx : l'encodeur bf16 (16,8 Go) est sur
  la carte pour le prompt et le négatif pendant que le DiT (10,2 Go) y reste, plus
  l'encodage VAE fp32 de l'image et les activations ; le pic est inhérent à l'offload
  « encodeur seul » en bf16 sur 32 Go (lot 1 : 27 Go à 1 Mpx). Les rendus suivants
  tiennent à 15 Go (prompt en cache). Piste (lot 3b) : pendant l'encodage d'un prompt
  neuf, sortir le DiT en RAM (10 Go, ~1 s par sens sur PCIe 5) avant de monter
  l'encodeur, et le remonter après : pic attendu ~19 Go, coût ~2 s par prompt neuf.
  Alternative sans code : l'encodeur nf4 (plafond 21 Go, verdict sur images en attente).
- **Draft** : le texte tient à 12 steps (24,8 s avec le prompt neuf, soit ~18 s à
  chaud, contre 85 s à 50). Proposition : remplacer « Draft (25 steps) » par
  « Draft (12 steps) » [12, 3.0], aide « brouillon : composition et détails peuvent
  bouger, textes fiables ». Décision de Mike.
- Refine 12 contre 36 steps : non jugé (images non reçues) ; 7,8 contre 19,6 s.

### Ménage des références à Ideogram (04/11/2026, demande de Mike)

- **Retirés** : `cz_ideogram.py` (ses marqueurs de famille Ideogram, `llm_cond_proj`...,
  et la largeur 4608 des LoRAs sont repris dans `cz_moss`, qui défère à `cz_flux3` pour
  FLUX : un fichier Ideogram reste refusé en nommant crispz-ideogram),
  `tests/test_ideogram_engine.py`, `test_ideogram_emulated.py`, `test_ideogram_e2e.py`,
  `tests/ideogram_tiny.py` ; la projection d'aperçu Ideogram de `cz_pipeline`
  (`_preview_proj_for` rend None : MOSS range ses tokens dans l'ordre FLUX, la matrice
  FLUX.2 s'applique telle quelle) et sa partie de `test_latent_preview` ; le profil
  `ideogram` de `cz_core.MODEL_PROFILES` ; les alias CLI `--ideogram-model` /
  `--ideogram-transformer`.
- **Renommés crispz-moss / MOSS-Image** : tous les scripts (`run`, `install`, `update`,
  `boot_check`, `xyz_example`, `civitai_index*`, .bat et .sh), les en-têtes de modules,
  l'Asset Browser (titre, user agent CivitAI), `fooocus_extra.json`, `SECURITY.md`,
  `config_modification_tutorial.txt`, `styles/SOURCE.txt`, `wildcards/example.txt`,
  les commentaires de `requirements*.txt`, les aides de l'UI (presets, ratio, guidance,
  sampler, offload, dossier de checkpoints, transformer, CivitAI), les messages du CLI,
  du protocole, du handoff et de `cz_vector` / `cz_prompt`.
- **Vérification d'installation** : `install.*`, `update.*` et `boot_check.bat`
  testaient les blocs modulaires FLUX.3 de diffusers ; ils importent désormais
  `transformers.Qwen3_5Model` et le VAE FLUX.2 de diffusers (la pile que MOSS exige).
- **Sélecteur « Text encoder »** : resté grisé avec l'aide Ideogram alors que
  `cz_pipeline` applique un encodeur de remplacement depuis le lot 1 ; rendu actif et
  câblé (`_ui_set_text_encoder`, refus nommé d'une autre taille).
- `README.md` réécrit pour MOSS (licence Apache 2.0 en tête, chaîne des licences, ce que
  fait l'app, mesures des lots 1 à 3, A/B, limites) ; `README_CLI.md` adapté ; `FORK.md`
  réécrit (base crispz-ideogram, tableau des différences), l'ancien gardé en
  `FORK-ideogram.md`.
- Test : `test_no_visible_flux3_or_ideogram_wording_in_the_ui` parcourt l'UI construite
  et n'admet « Ideogram » que pour nommer crispz-ideogram ou les LoRAs refusées.
- Restent, à dessein : `CLAUDE.md`, `CHANGELOG.md`, `FORK-*.md` (historique) ; les
  refus qui nomment crispz-ideogram ; `tools/gpu_lot1_check.py` (A/B contre ses affiches).
- Suite complète : **66 fichiers sur 66** (3 fichiers Ideogram retirés) ; smoke 22/22 ;
  `check_config_keys` OK.

### Catalogue de personnages et « Send to crispz-moss » (04/11/2026, demande de Mike)

- **`cz_catalog.py`** : le MOSS-Image Character Catalog (`hide1/MOSS-Image-Character-Catalog`,
  Space et dataset HF ; lu le 04/11/2026) : 33 935 personnages (3 953 séries) et 314 styles
  rendus par MOSS-Image, chacun avec `copy_prompt`, `copy_tags`, `reuse_generation_prompt`
  (plein cadre : un suffixe de cadrage identique partout, stocké une fois par manifeste)
  et des facettes (cheveux, yeux, genre, tenue). `catalog.json.gz` (7 Mo, MIT pour les
  données de l'auteur) téléchargé UNE fois sur le bouton de l'UI, ligne `CATALOG:` dans le
  log, rangé dans `cache/moss/character_catalog.json.gz`. Les images restent sur le
  dataset HF : vignettes chargées par le navigateur seulement si la case est cochée
  (préférence `catalog_remote_thumbs`, off). Avis de droits de la NOTICE du catalogue
  (personnages, œuvres et images : IP tierces, pas pour un livrable client ; « non curé »)
  affiché dans l'accordéon, le bandeau du navigateur et chaque fiche.
- **Asset Browser** : deux onglets Characters et Styles (`_index/characters.json` 30 Mo,
  `styles.json`), dossiers latéraux = série (copyright), recherche sur nom, tags, prompt ;
  cartes paginées par 600 (34 000 divs d'un coup, non) ; fiche : nom, série, facettes,
  prompt, tags, prompt de génération, lien vers l'image sur HF. Manifestes réécrits à
  l'ouverture du navigateur s'ils sont absents ou périmés (`ensure_index`, méta
  `_index/catalog.meta.json`). Autocomplétion : `tags/moss_catalog.csv` (38 137 lignes :
  personnages catégorie 4 avec leur `source_count`, séries 3, styles 1), lu au prochain
  lancement comme tout CSV de `tags/`.
- **« Send to crispz-moss »** : depuis l'Asset Browser (son propre onglet), un bouton
  envoie un texte dans la zone de prompt de l'app : prompt d'une sortie (ou prompt +
  négatif), triggers d'une LoRA (ajoutés), **prompt d'un exemple CivitAI** d'une LoRA
  (remplace : pour essayer la LoRA sur son exemple), prompt / tags / prompt de génération
  d'une fiche du catalogue. Canal `BroadcastChannel('crispz-moss')` entre les deux onglets
  (même origine), `localStorage` en repli ; l'app (`CZ_JS`) écrit dans `#cz_prompt` ou
  `#cz_neg` et déclenche l'événement `input` pour Gradio. Rien ne quitte la machine.
- Vérifié dans Chromium avec le vrai catalogue : 600 cartes, 33 935 au compteur, 3 954
  dossiers, recherche « hatsune », fiche complète, envoi reçu par le canal et par
  localStorage ; puis sur deux onglets (app factice avec `CZ_JS` + navigateur) : remplace,
  puis ajoute, événement `input` compté.
- Tests : `tests/test_catalog.py` (7 : formes, suffixe, vignettes on / off, manifestes et
  fraîcheur, CSV, téléchargement mocké et loggé, pont dans le SPA et dans `CZ_JS`,
  accordéon dans l'UI). Suite : **67 fichiers sur 67** ; smoke 22/22.
- **Les rendus en local** (demande de Mike « récup les render aussi ») : deux boutons dans
  l'accordéon, `download_renders("thumbnails")` et `("images")` : `snapshot_download` du
  dataset HF (`allow_patterns`, 8 transferts parallèles, reprise : un second clic ne prend
  que ce qui manque, `hf_token` de la config s'il existe) dans
  `cache/moss/character_catalog/`, servi par l'app (`allowed_paths` au lancement et à la
  volée). Les cartes prennent le fichier local d'abord, l'URL HF seulement avec la case,
  sinon rien ; le manifeste dit les comptes sur disque et l'index se sait périmé quand
  des fichiers arrivent. Mesuré le 04/11/2026 sans token : vignettes 280 px, 7,8 Ko
  chacune (34 249 fichiers, environ 270 Mo), pleines images 54 Ko (environ 1,8 Go) ;
  10 à 14 fichiers par seconde en anonyme, soit environ une heure par lot : dit dans
  l'UI. Test : `test_renders_on_disk_come_first` (hub mocké). Suite : 67 fichiers sur 67.

### Lot 3b (05/11/2026) : placement du draft, sortie du DiT pendant l'encodage, Draft 12 steps

Les trois points ouverts du premier passage GPU du lot 3, lancés sur « passe au step
suivant » de Mike (04/11/2026), livrés sur `wip/moss-engine`, mesure GPU à faire.
- **Placement du draft en outpaint et reframe** : `prepare_prompt(..., place=(x0, y0,
  x1, y1))` reporte les boîtes d'un draft dans le sous-rectangle du canevas où l'image
  source est collée (mise à l'échelle dans ce rectangle, jamais étirée sur le canevas) et
  force « No other text. » dans le résumé ; un prompt en texte libre reçoit le même
  garde-fou (ajouté une fois, dit). `outpaint_directions` calcule le rectangle (marges
  gauche et haut, puis l'échelle de `_cap_work_res`), `reframe` passe `(ox, oy, ox + rw2,
  oy + rh2)` ; les deux le donnent à `_qwen_call` par `layout_box`, consommé avant le
  cache d'embeddings. Inpaint et refine gardent `fit` (l'image entière).
  `gpu_lot3_check.py` outpaint et reframe avec le draft placé par défaut, `--plain` pour
  l'ancien comportement. À juger sur les mêmes rendus : plus de faux textes ?
- **Sortie du DiT pendant l'encodage** (`moss_encode_swap`, défaut true, env
  `MOSS_ENCODE_SWAP`) : sous l'offload « encodeur seul », `_encoder_in` gare le DiT en RAM
  avant de monter l'encodeur, `_encoder_out` (fin d'encodage, ou le pré-hook du premier
  forward du DiT) le remonte. Les deux ne partagent plus jamais la carte : pic attendu
  ~19 Go au lieu de 27 à 30 sur un prompt neuf, coût ~1 s par sens par prompt NEUF (un
  prompt en cache ne bouge rien). Sans effet avec l'encodeur nf4 (résident, jamais
  déplacé). Le magic prompt local et le Describe local en profitent aussi : 10 Go
  libérés pendant la génération. `_execution_device` reste cuda pendant la garde (le VAE
  y est, diffusers préfère un composant hors CPU). Dit une fois dans le log.
  Test : `test_encode_swap_parks_the_dit_while_the_encoder_is_on_the_card` (ordre exact
  des déplacements, image identique, rien ne bouge sur un prompt en cache).
- **Preset « Draft (12 steps) »** [12, 3.0] remplace « Draft (25 steps) » (cz_ui,
  config-sample, aide) : le texte tenait à 12 steps sur la 5090 (24,8 s contre 85 s).
  ATTENTION : un `config.txt` local qui a copié `performance_presets` garde l'ancien.
- Tests : +3 (`test_prepare_prompt_places_a_draft_on_the_source_image`,
  `test_outpaint_and_reframe_place_the_draft_and_guard_the_new_areas`, le swap) ; le test
  d'offload existant passe `swap_dit=False` (il vérifie que le DiT ne bouge pas sans
  swap). Suite complète : **67 fichiers sur 67** ; smoke 22/22 ; `check_config_keys` OK
  (123 clés).
- À faire chez Mike : `git pull`, carte libre, `.venv\Scripts\python tools\gpu_lot3_check.py
  --skip-draft` ; regarder le pic du premier img2img (attendu ~19 Go au lieu de 30), et
  `3_outpaint_right.png` / `4_reframe_16x9.png` (zones neuves sans texte inventé ?).

### Lot 4 (05/11/2026) : chargeur de LoRA, livré sur `wip/moss-engine`, mesure GPU à faire

Lancé sur « passe au step suivant » de Mike, après le lot 3b.
- **Relevé des vraies LoRAs publiées** (en-têtes safetensors lus le 04/11/2026), quatre
  dialectes :
  | fichier | clés | alpha | particularité |
  |---|---|---|---|
  | `lodestones/MOSS-Image-SFT-8-steps-dmd` `lora.safetensors` (756 tenseurs, fp32) | kohya `diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}.lora_down/lora_up.weight` + `.alpha` | 128 par module, rang 128 | la recette DMD complète remplace aussi `time_embedding` et échantillonne avec sa boucle (lot 3) : ici la LoRA seule |
  | `Comfy-Org/MOSS-Image` `loras/moss_image_color_depth_control` (526) | peft `layers.N.proj.*.lora_A/B.weight`, `text_embedder.blocks.N.*` | « alpha » global 64 dans les métadonnées | **remplace `video_embedder` (384 canaux d'entrée)** : un modèle de CONTRÔLE qui attend des cartes couleur et profondeur, refusé par nom |
  | `Comfy-Org/MOSS-Image` `loras/moss_image_edit_lora_rank_64_bf16` et `chinoll/Moss-Image-Edit` `adapter_model` (504) | `layers.N.proj.*.lora_A` / `lora_B` SANS `.weight` | 64 par module dans `__metadata__.adapter_config` (format `moss-edit-lora`) | contrat `ref-prefix-only-t0__target-text-original-moss-v1` : les références entrent dans la séquence, voir l'étude plus bas |
  | `MATLOWAI/MOSS-Image-SFT-DeBand` (3 fichiers, 40 à 84 tenseurs) | ComfyUI `diffusion_model.layers.N.<module>.diff` : différences de poids COMPLÈTES, normes (`norm1`, `norm2`, `q_norm`, `k_norm`) comprises | aucun | `proj.qkv.diff` fusionné [9216, 3072] (variante `_splitqkv` séparée) ; « remet » les couches 37 à 41 (puis 32 à 36) aux poids Base |
- **`cz_moss_lora.py`** : lit les quatre dialectes (préfixes `diffusion_model.`,
  `transformer.`, `base_model.model.`, `model.` retirés ; alpha par tenseur, sinon par
  module via `adapter_config`, sinon global, sinon rang), scinde le `layers.N.proj.qkv`
  fusionné de ComfyUI en q, k, v (le `text_embedder.blocks.N.qkv` est fusionné dans le
  modèle, gardé tel quel). Application par **hooks forward** sur les Linear du DiT
  (`y += scale * x A^T B^T`, `y += w * x D^T` pour un diff, décalage de sortie pour un
  `diff_b`), delta en place sur les poids des normes avec restauration à la décharge ;
  tenseurs enregistrés en buffers des modules (ils suivent le DiT entre la carte et la RAM
  au swap du lot 3b). Poids de base jamais touchés, re-pondération en place sans
  recharger (`rescale`, sauf quand un fichier porte des diffs de normes : réappliqué).
  **Clés inconnues et modules absents comptés et dits** dans le log, jamais ignorés en
  silence ; un fichier qui n'applique rien le dit. Refus nommés : LoRA de contrôle
  (remplace une couche), fichier sans tenseur LoRA.
- `MossPipe.set_loras([(chemin, poids)])` / `applied_loras` ; `load_lora_weights` refuse
  en renvoyant vers `set_loras`. `cz_pipeline._apply_loras` route un MossPipe vers
  `_apply_moss_loras` (fichiers refusés SAUTÉS avec la raison, absents ignorés, même
  jeu = rien à refaire) ; `_clear_loras` vide le jeu ; `_lora_unsupported` laisse passer
  une LoRA MOSS et nomme la LoRA de contrôle. Les menus LoRA de l'UI listent donc les
  fichiers MOSS comme applicables.
- Tests : `tests/test_moss_lora.py` (9 : classification, kohya alpha/rang, qkv fusionné
  scindé + clés inconnues dites, moss-edit-lora à alpha par module, diffs sur Linear,
  normes et biais avec décharge qui restaure, refus contrôle et fichier vide, rescale et
  poids 0 = identité, rendu qui change puis revient sur le MossPipe, routage
  cz_pipeline). Tests hérités mis à jour (le refus « lot 4 » devient un passage de garde,
  la LoRA de contrôle refusée). Suite : 67 fichiers sur 67 + 1 nouveau.
- `tools/gpu_lot4_check.py` : baseline (affiche, 50 steps), DeBand L37-41 à 1,0 (même
  seed : rayures verticales parties ?), LoRA DMD seule à 8 steps CFG 1 et 12 steps CFG 3
  (expérience : sans l'embedding de temps ni la boucle DMD), Draft 12 steps ; fichiers
  téléchargés dans `loras/` (Apache 2.0 tous deux), temps et pics dans `bench/lot4/`.
- **Point 3 du cahier (édition par références, `Moss-Image-Edit`)** : la LoRA se charge
  (dialecte `moss-edit-lora`), mais sans la construction de séquence des nœuds ComfyUI
  (références en préfixe à t = 0) elle ne sert à rien. Le code des nœuds est GPLv3 : lu
  pour comprendre le contrat, jamais à vendoriser (l'app est CC BY-NC) ; ce qui suit est
  une description, le code sera réécrit depuis elle sur le DiT vendorisé (Apache 2.0).
- **Étude des nœuds `Chenkin-x/ComfyUI-Moss-Image-Edit`** (`nodes.py`, `native_edit.py`,
  README, lus le 05/11/2026 ; le code source du dépôt d'entraînement et la fiche HF des
  poids n'ont pas été lus). Contrat `ref-prefix-only-t0__target-text-original-moss-v1` :
  | point | ce que font les nœuds |
  |---|---|
  | références | l'image source D'ABORD, puis `reference_0..N` (le README dit 4 au plus, le code ne plafonne pas) ; chacune réduite à 512 px de grand côté (`reference_max_edge`, multiples de 16, lanczos, jamais recadrée), encodée UNE fois par le VAE FLUX.2 |
  | séquence | `[ref_1 ... ref_N, cible, texte]` : les références passent par le MÊME `video_embedder` que la cible et sont mises en PRÉFIXE (le schéma du README dit `[texte, ref, cible]`, le code dit préfixe) |
  | temps | modulation des références avec `time_embedding(t = 0)` (image propre), cible et texte avec le vrai t. Le `_Block.forward` vendorisé applique UNE modulation à toute la séquence : il faut une modulation par segment |
  | RoPE | référence i : coordonnée 0 = i + 1 (la cible et le texte ont 0), y et x centrés comme la cible ; la coordonnée 1 des références n'est pas vérifiée (la cible y porte `text_length`) |
  | attention | les requêtes des références ne regardent que les clés du préfixe ; cible et texte regardent TOUT. Réalisable sans masque : deux appels SDPA par bloc (préfixe sur lui-même, corps sur la séquence entière) |
  | sortie | seule la tranche cible passe par `final_norm_video` / `final_linear_video` ; les références ne sont ni mises à jour ni rendues ; recalculées à chaque forward (pas de cache K/V dans le port ComfyUI, le dépôt d'origine en décrit un) |
  | échantillonnage | depuis un latent vide à la taille demandée, scheduler normal ; la LoRA est un patch à l'exécution (pas de fusion), force 0 = modèle de base |
  | texte | l'instruction est encodée par le nœud ComfyUI de base « Text Encode MOSS-Image Edit » (hors dépôt) : `target-text-original-moss` suggère le texte tel quel, non vérifié |
  | licences | nœuds et dépôt d'entraînement GPLv3 ; **poids de la LoRA : aucune licence dite** dans les README (fiche HF à lire) ; les LoRAs Comfy-Org `moss_image_edit_lora_rank_64` suivent la même métadonnée |
  **Faisabilité dans crispz-moss** : tout tient dans notre code, sans toucher au fichier
  vendorisé : une passe d'édition dans `cz_moss` (embedding des références, modulation
  à t = 0 sur le préfixe, RoPE avec index de référence, deux SDPA par bloc, tranche
  cible), une centaine de lignes ; le chargeur du lot 4 applique déjà la LoRA sur toutes
  les positions, préfixe compris. À régler avant d'écrire, en lisant la fiche HF et le
  dépôt d'entraînement (lecture seule) : la coordonnée 1 des références, si les
  références se voient entre elles ou chacune seule, la construction du texte, et la
  licence des poids. Ordre de travail proposé (lot 4b, après le verdict GPU du lot 4) :
  passe d'édition + test CPU sur le miniature (zéro référence = image identique au
  forward de base, forme de sortie, références jamais écrites), puis
  `gpu_lot4_check.py --edit` avec la LoRA Comfy-Org (bf16, rang 64) sur l'affiche du
  lot 2 (« remplace le pain par un croissant », une référence), onglet « Edit
  (references) » réactivé seulement si le rendu tient, avec la licence des poids
  affichée « non précisée » tant qu'elle ne l'est pas.
- À faire chez Mike : `git pull`, carte libre, `.venv\Scripts\python tools\gpu_lot4_check.py`
  (télécharge 2 LoRAs, 1,3 Go) ; juger `1_deband` contre `0_baseline` sur les aplats, et
  les textes des rendus DMD.

### Passage GPU des lots 3b et 4 (05/11/2026, Mike) : swap mesuré, LoRAs lues sur images

`gpu_lot3_check.py --skip-draft` puis `gpu_lot4_check.py`, à la suite, 07:33 à 07:46,
offload 'model' encodeur seul, encodeur bf16, swap du DiT actif.

| lot 3b | temps (avant 3b) | pic (avant 3b) | lecture |
|---|---|---|---|
| img2img 0,3, premier rendu | 93,7 s (71,0) | **23,5 Go (30,0)** | le swap tient sa promesse sur le pic, pas les 19 Go espérés |
| img2img 0,6 | 54,7 s (52,3) | 23,5 (réservé, non rendu) | |
| inpaint | 46,7 s (43,4) | 23,5 (idem) | |
| outpaint à droite, draft placé | 48,2 s (43,5) | 20,6 (27,2) | images non reçues : faux textes partis ? à juger |
| reframe 16:9, draft placé | 49,5 s (39,7) | 20,1 (14,3) | prompt neuf (draft placé) : l'encodeur est revenu, d'où le pic |
| refine 0,3 à 12 / 36 steps | 8,2 / 20,6 s | 15,8 | images non reçues (déjà au lot 3) |

Lecture : le pic du premier img2img passe de 30 à 23,5 Go. Il reste au-dessus des 19 Go
visés parce que l'encodeur (16,8 Go) était ENCORE sur la carte pendant l'encodage VAE de
l'image (1,9 Mpx en fp32, plusieurs Go d'activations) : l'ordre était négatif, puis VAE,
puis prompt, puis premier forward du DiT (où l'encodeur repart). Corrigé au lot 4b :
quand une image doit être encodée, le prompt est encodé d'abord, l'encodeur repart (le
DiT revient), le VAE encode ensuite ; pic attendu 17 à 19 Go, à remesurer. Le coût du
swap sur cette machine : environ 5 à 10 s par prompt NEUF (le DiT de 10 Go en RAM
pageable et retour, plus que la seconde par sens estimée), visible sur les 4 rendus à
prompt neuf (+ 3 à 22 s). Un prompt en cache ne coûte rien. `moss_encode_swap: false`
rend les 30 Go de pic contre ces secondes : au choix de Mike.

| lot 4 (affiche preset, 1152x1632, seed 4242) | temps | pic | lecture sur image |
|---|---|---|---|
| 0 baseline 50 steps | 110,1 s | 21,1 Go | affiche plate propre, « HEADLINE » et « DATE · PLACE » exacts, polo bleu, aucun banding visible |
| 1 DeBand L37-41 à 1,0 | 99,8 s | 22,1 | même composition, textes exacts, mais le STYLE change : grain de papier fin sur toute l'image, t-shirt au lieu du polo, jaune plus froid |
| 2 LoRA DMD seule, 8 steps, CFG 1 | 17,9 s | 22,8 (prompt ré-encodé) | **image propre et complète**, textes exacts, pull vert, visage tourné ; aucun artefact de sous-échantillonnage |
| 2 LoRA DMD seule, 12 steps, CFG 3 | 24,1 s | 16,9 | textes exacts, quasi identique au Draft de base ci-dessous |
| 3 Draft base 12 steps | 21,7 s | 16,8 | textes exacts ; à l'œil la même image que la LoRA DMD à 12 steps |

Lecture :
- **DeBand** : rien à corriger sur ces affiches (pas de banding au départ) et la LoRA
  change le rendu (elle remplace les 5 derniers blocs par ceux de Base : un grain de
  papier apparaît, le vêtement change). À garder opt-in, jamais par défaut ; utile
  seulement si des rayures verticales apparaissent sur un aplat.
- **LoRA DMD seule à 8 steps, CFG 1** : contre toute attente (sans l'embedding de temps
  ni l'échantillonneur de l'auteur), l'image est propre, les textes exacts. Un seul
  forward par step : le rendu réel est d'environ 8 à 10 s (les 17,9 s comptent un
  ré-encodage du prompt, pic à 22,8 Go), contre 21,7 s pour le Draft de base à 12 steps
  (deux forwards par step). À 12 steps CFG 3 la LoRA n'apporte rien sur le Draft de base.
  **Proposition** : un preset « Draft (8 steps, DMD LoRA) » [8, 1.0] qui télécharge la
  LoRA (1,2 Go fp32, Apache 2.0) une fois, comme le preset d'édition, et la retire pour
  les rendus normaux. Décision de Mike ; à mesurer sur 4 seeds avant (un rendu).
- Les 5 images sont des rendus différents du même seed : chaque LoRA change la
  composition (vêtement, orientation du visage), c'est attendu (un rendu à 50 steps ne
  « converge » pas vers celui à 12).

### Lot 4b (05/11/2026) : édition par références, livré sur `wip/moss-engine`, mesure GPU à faire

Lancé sur « passe au step suivant » de Mike. Les quatre inconnues de l'étude du lot 4
levées en lecture seule (fiche HF, dépôt d'entraînement `Chenkin-x/Moss-Image-Edit`,
`comfy/ldm/moss_image/model.py` et `comfy_extras/nodes_moss_image.py` de ComfyUI, le
05/11/2026) : RoPE des références `(i + 1, 0, y centré, x centré)` (cible : `(0,
longueur du texte, y, x)`, texte : `(0, j, 0, 0)`) ; les références se voient TOUTES
entre elles (un seul préfixe) ; l'instruction passe telle quelle dans le gabarit de chat
(pas de système, pas de légende, l'encodeur ne voit pas les images) ; licence des poids :
AUCUNE dans la fiche HF (code GPLv3 seulement). Entraînement : une référence = l'image
source, cible de même taille ; inférence : `max_edge` 1024 par défaut (256 dans son
json, 512 dans le nœud ComfyUI), 4 références au plus, 50 steps, CFG 4, négatif vide,
références présentes dans les deux branches du CFG. LoRA : `proj.{q,k,v,o,ffn_up_gate,
ffn_down}`, échelle alpha / rang.
- **`cz_moss.EditPrefix`** (notre code, rien du GPLv3 repris, le fichier vendorisé
  intact) : références encodées comme la cible (`_encode_latents`, même `video_embedder`),
  préfixe modulé à t = 0, passe du préfixe UNE fois par appel (attention préfixe seul,
  toutes les couches) avec les clés tournées et valeurs de chaque couche gardées (cache
  K/V, `EDIT_CACHE_MAX_GB` 6 Go, sinon recalcul à chaque forward, dit) ; le corps
  [cible, texte] attend sur [préfixe, cible, texte] avec deux SDPA par bloc (préfixe
  mis en tête des clés) ; tête de sortie sur la tranche cible. Sans référence : la passe
  vendorisée **bit pour bit** (test, couches sautées comprises). Les hooks LoRA du lot 4
  jouent sur le préfixe aussi (même Linear).
- `MossPipeline.__call__(reference_latents=[...])` ; `_cz_before_dit` (posé par
  MossPipe) ramène le DiT garé avant la passe du préfixe (le pré-hook du module ne voit
  pas cette passe). `MossPipe` : `image` sans strength ni masque = édition ; **refus
  NOMMÉ sans LoRA d'édition appliquée** (`edit_loras_applied`, métadonnée
  `moss-edit-lora` lue par `cz_moss_lora.is_edit_lora`) ; Ref 1 = source, sortie à sa
  taille sauf taille explicite ; références ramenées à `moss_edit_ref_max_edge` (1024)
  sur le grand côté, jamais agrandies, grille 16 ; 4 au plus (dit) ; Stop joue.
- **Texte d'abord, VAE ensuite** : quand une image doit être encodée (img2img, inpaint,
  édition), le prompt est encodé avant, l'encodeur repart (DiT de retour) et SEULEMENT
  ensuite le VAE encode : le pic de 23,5 Go du lot 3b (encodeur + VAE ensemble) devrait
  tomber vers 17 à 19 Go. Une visite de l'encodeur pour négatif + prompt.
- `cz_edit_loras` : le catalogue devient le seul preset « Moss-Image-Edit »
  (`Comfy-Org/MOSS-Image` `loras/moss_image_edit_lora_rank_64_bf16.safetensors`, poids
  1,0, 4 entrées, exemple de l'auteur, `licence` « aucune licence dite par l'auteur »
  affichée dans l'onglet, le log, les caps). `edit_ready()` = fichier sur disque.
  `cz_pipeline._apply_moss_edit_loras` : jeu de base + jeu d'édition par le chargeur du
  lot 4 ; sans LoRA d'édition dans les deux jeux, le preset est résolu (téléchargé une
  fois, ~300 Mo) et appliqué pour l'édition, dit une fois avec la licence ; retiré au
  prochain rendu texte vers image ; fichier refusé = erreur dure, rien de rendu.
  `outpaint_lora` (voie LoRA de crispz-flux3) refuse par nom sur MOSS.
- `cz_protocol` : `OMNI_SUPPORTED` True, `supports.edit` / `refs` = LoRA sur disque,
  `MAX_REFS` 4, op `edit` refusée (code 3) avec la marche à suivre tant que la LoRA
  manque, refs ignorées avec la raison. UI : onglet « Edit (references) » et mode
  d'entrée de retour, texte réécrit (Ref 1, références en mots, LoRA et licence, 1024
  px, 50 steps CFG 4 chez l'auteur), bouton sans emoji, `check_omni_available` dit si la
  LoRA est là. `config-sample` : `moss_edit_ref_max_edge`, aides mises à jour.
- Scheduler : `load_scheduler` ne passe que `num_train_timesteps`, `shift`,
  `training_shift` (le json du repo porte les clés de la classe mère de diffusers :
  avertissement « not expected and will be ignored » à chaque chargement, sans effet).
- Tests : `tests/test_moss_edit.py` (10 : égalité bit pour bit sans référence, cache =
  recalcul et indépendant de t, références jamais écrites, coordonnées RoPE contre la
  table vendorisée, tailles, refus nommé, taille source gardée et références plafonnées
  à 4, Stop, generate_omni qui applique le preset puis le retire, caps du protocole,
  config du scheduler sans avertissement) ; `test_edit_lora_presets` réécrit (catalogue
  MOSS, téléchargement HF mocké, LoRA MOSS d'édition qui passe la garde) ;
  `test_protocol` (présence de la LoRA simulée), `test_local_tools` (onglet offert),
  `test_moss_engine`, `test_moss_wiring`, `test_flux3_wiring` (refus de l'outpaint LoRA).
- `tools/gpu_lot4b_check.py` : sur l'affiche du lot 2, édit 1 référence (croissants) à
  CFG 4 et 3, 2 références (le logo du lot 2 à poser en haut à droite), références à
  512 px contre 1024, fond du logo (exemple de l'auteur), contrôle sans LoRA refusé ;
  `bench/lot4b/report.md`.
- À faire chez Mike : `git pull`, carte libre, `.venv\Scripts\python tools\gpu_lot4b_check.py`
  (télécharge la LoRA, ~300 Mo) ; juger : le changement demandé et LUI SEUL (textes et
  reste intacts), le logo de la 2e référence, 512 contre 1024 ; relancer
  `gpu_lot3_check.py --skip-draft` pour le pic du premier img2img (attendu 17 à 19 Go).

### Premier passage GPU du lot 4b (05/11/2026, Mike) : l'édition marche, les textes et la 2e référence non

`tools/gpu_lot4b_check.py`, 08:24 à 08:39, LoRA `moss-image-edit.safetensors@1`
(téléchargée par le preset), offload 'model' encodeur seul, 50 steps, références lues à
1024 px de grand côté (l'ancien défaut) :

| étape | temps | pic | lecture sur image |
|---|---|---|---|
| 0 édit 1 réf, CFG 4 (affiche 1152x1632 lue en 720x1024) | 131,2 s | 23,1 Go | **croissants posés dans le panier, composition et style gardés**, fond blanchi ; textes REDESSINÉS et abîmés : « GRAND OPENNAJ », « SATUIDAY 9 AM · RUE BU PAIN » |
| 1 idem, CFG 3 | 110,6 s | 18,0 | croissants OK ; textes pires : « GRANOOPENENUG », « SATUDAY 9 AM RUE BLLKIN » |
| 2 édit 2 réfs (affiche + logo « BEAN THERE ») | 145,6 s | 21,1 | **fuite de la 2e référence** : un badge rond « COFFEE COFFEE » inventé en haut à droite, le texte « BEAN THERE » et le symbole du logo PLAQUÉS sur le panier, bandes noires haut et bas (le cadrage carré du logo) ; « GRAND OPENING » exact cette fois, sous-titre abîmé |
| 3 édit 1 réf lue à 512 px | 265,8 s | 16,5 | **source perdue** : une autre affiche (photo de pains, charabia « GIR'SK IOOSTILINS » et six lignes inventées) |
| 4 fond du logo (1024, exemple de l'auteur) | 192,4 s | 21,1 | **parfait** : symbole et « BEAN THERE » intacts au pixel près, plage et mer derrière |
| 5 contrôle sans LoRA | | | refusé par nom, rien rendu |

Lecture :
- **Le contrat tient** : la passe de préfixe fonctionne sur les vrais poids (l'exemple de
  l'auteur est reproduit à l'identique, les croissants remplacent le pain sans toucher à
  la composition). Critère « un edit aboutit » atteint.
- **Les textes sont redessinés, pas copiés**, et leur fidélité dépend de la résolution à
  laquelle la source entre : à 1024 (affiche réduite à 720x1024) des lettres cassent, à
  512 la source est perdue. La LoRA a été entraînée sur des PAIRES de même taille
  (source = l'unique référence, cible) : lire la source à sa propre taille est ce que le
  modèle a vu. **Corrigé** : `moss_edit_ref_max_edge` vaut désormais 2048 (la source à
  sa taille, l'écart ne peut venir que d'une image au-delà de 2048), aide et onglet
  réécrits ; le script compare « taille d'origine » contre 1024. CFG 4 (l'auteur) tient
  mieux les textes que CFG 3 : dit dans l'onglet.
- **Plusieurs références : expérimental**, dit dans l'onglet, le README et le log (une
  fois) : la LoRA n'a vu qu'UNE référence à l'entraînement ; la deuxième fuit (contenu et
  cadrage). La capacité reste (le contrat l'autorise, l'auteur l'annonce), le garde-fou
  est l'avertissement.
- **Temps** : 131 s pour l'affiche contre 110 s en texte vers image (le cache K/V du
  préfixe fait son travail : +20 %). Les 265 s du rendu à 512 (préfixe PLUS petit) et les
  192 s du logo 1024 (42 s en texte vers image) sont **inexpliqués** : ni le préfixe ni le
  pic de VRAM ne les justifient ; les deux sont les derniers rendus du passage. Piste :
  un autre processus sur la carte, ou de la mémoire paginée. Le pipeline logue désormais
  le temps de la passe du préfixe et la taille du cache (ligne `edit prefix:`), repris
  dans le rapport ; à relancer en surveillant `nvidia-smi`.
- Suite : **69 fichiers sur 69** ; smoke 22/22.
- À faire chez Mike : `git pull`, carte libre, `tools\gpu_lot4b_check.py` (références à
  leur taille) ; juger les textes de l'étape 0 contre ce premier passage et contre
  l'étape 3 (1024), et les temps.

### Deuxième passage du lot 3b (05/11/2026, Mike, 08:52) : outpaint et reframe lus, pic inchangé

`gpu_lot3_check.py --skip-draft` après le réordonnancement texte / VAE du lot 4b :
- **Pic inchangé** : 23,48 Go au premier img2img, le même chiffre à la décimale qu'avant le
  réordonnancement. Le pic n'est donc PAS la phase VAE mais l'encodage du texte lui-même :
  l'encodeur bf16 (16,8 Go) plus ses temporaires sur un prompt layout long (noyaux GDN en
  repli torch, float32), 6 à 7 Go. Le DiT est bien garé (sinon 30 Go). C'est le plancher de
  l'encodeur bf16 sur un prompt neuf ; en dessous, seule la voie nf4 (21 Go constants,
  verdict sur images toujours en attente). Le réordonnancement a gagné 14 s sur le premier
  rendu (79,7 s contre 93,7), le reste est identique. Pour trancher au prochain run, le
  pipeline écrit une ligne `vram:` par appel avec image (réservé et pic après le texte,
  puis après l'encodage VAE).
- **Outpaint à droite (draft placé)** : l'affiche d'origine intacte, textes exacts ; dans la
  bande neuve plus aucun charabia, mais « ING » en haut et « | PAIN » en bas, à la hauteur
  exacte du titre et du sous-titre. **Reframe 16:9 (draft placé)** : l'affiche centrée
  intacte ; dans les bandes latérales « GRAN » / « NING » et « SATURD » / « DU PAIN » en
  lettres géantes. Lecture : le modèle rend les textes du layout à l'échelle du CANEVAS
  (« GRAN[D OPE]NING » dont le milieu est recouvert par la zone gardée), les boîtes placées
  ne suffisent pas à le confiner. L'init des bandes n'y est pour rien (strength 1,0 :
  bruit pur). **Correctif** : chaque bande neuve est DÉCLARÉE dans le layout comme un
  élément `object` à sa boîte (« plain empty background ..., no text, no letters, no
  objects », `cz_moss.MARGIN_CAPTION`, `margins_of`), et un prompt texte libre reçoit la
  phrase équivalente qui nomme les côtés. Tests : `test_caption`, `test_moss_wiring`.
  À remesurer sur les mêmes rendus.

### Troisième passage du lot 3b (05/11/2026, Mike, 09:25) : img2img lus, pic toujours 23,48

`gpu_lot3_check.py --skip-draft` avec les bandes déclarées : premier img2img 74,6 s (93,7
puis 79,7 aux passages précédents), pic 23,48 Go pour la troisième fois à la décimale
(le plancher de l'encodeur bf16 sur un prompt neuf, confirmé par la constance ; la ligne
`vram:` de la console tranche définitivement). Outpaint 46,6 s, reframe 46,1 s, refine
7,7 / 19,5 s : stables. Images reçues : les deux img2img (0,3 et 0,6) de l'affiche aux
croissants, identiques à l'œil, « GRAND OPENING » et « SATURDAY 9 AM · RUE DU PAIN »
exacts, composition intacte : l'img2img avec layout tient aux deux forces. Les rendus
d'outpaint et de reframe de ce passage (bandes déclarées) ne sont pas encore reçus.

### Deuxième passage GPU du lot 4b (05/11/2026, Mike, 09:38) : textes exacts à la taille d'origine

`tools/gpu_lot4b_check.py`, références lues à leur taille (défaut 2048), ligne
`edit prefix:` dans le rapport, 11 minutes en tout :

| étape | temps | pic | préfixe | lecture sur image |
|---|---|---|---|---|
| 0 édit 1 réf, CFG 4, affiche à 1152x1632 | 170,1 s | 23,95 Go | 7344 tokens, passe 1,0 s, cache 3,53 Go | **croissants dans le panier, « GRAND OPENING » et « SATURDAY 9 AM · RUE DU PAIN » EXACTS**, fond crème et style gardés |
| 1 idem, CFG 3 | 125,5 s | 20,4 | idem | **textes exacts aussi**, croissants un peu différents |
| 2 édit 2 réfs (affiche + logo) | 153,9 s | 23,3 | 11440 tokens, 1,7 s, 5,50 Go | textes de l'affiche exacts ; un badge café posé en haut à droite comme demandé (badge générique « COFEE / COFFEE », pas le logo « BEAN THERE »), mais le symbole et le texte du logo fuient encore sur le panier |
| 3 idem étape 0, référence lue à 1024 | 105,3 s | 18,0 | 2880 tokens, 0,3 s, 1,38 Go | rendu identique au premier passage (même seed) : « OPENNAJ », « SATUIDAY », « RUE BU PAIN » |
| 4 fond du logo 1024 | **62,3 s** (192,4 au premier passage) | 21,1 | 4096 tokens, 0,4 s, 1,97 Go | identique au premier passage : parfait |
| 5 contrôle sans LoRA | | | | refusé par nom |

Lecture :
- **Critère d'acceptation du lot 4b atteint sur une référence** : la source lue à sa
  taille rend les textes exacts à CFG 4 ET à CFG 3, l'édition demandée est faite et elle
  seule (fond, panier, typographie intacts). La même édition à 1024 casse les lettres :
  la règle « paires de même taille » de l'entraînement est confirmée sur images ;
  le défaut 2048 est le bon. Coût : 125 à 170 s pour 1,9 Mpx (texte vers image : 110 s),
  pic 24 Go, cache K/V 3,5 Go.
- **Deux références : toujours expérimental.** Le logo demandé arrive bien dans le coin
  (c'est l'instruction), mais redessiné en badge générique, et le symbole + le texte de
  la référence se plaquent encore sur le panier. L'avertissement reste ; pas d'autre
  correctif tant que la LoRA n'est entraînée qu'à une référence.
- **Les temps anormaux du premier passage étaient externes** : le fond du logo passe de
  192 à 62 s à réglages identiques, image identique au pixel ; la passe du préfixe
  coûte 0,3 à 1,7 s, le cache tient sous 6 Go. Rien à corriger dans le code.
- **Lot 4b VALIDÉ par Mike le 05/11/2026** (une référence) ; les rendus d'outpaint et de
  reframe du passage de 09:25 (bandes déclarées) ne sont pas encore reçus. Décision du
  preset « Draft (8 steps, DMD LoRA) » toujours ouverte.

### Lot 5 (05/11/2026) : outils locaux et passerelles, livré sur `wip/moss-engine`

Lancé sur « Lot 4b validé, passe au step suivant » de Mike.
- **Remove BG (rembg) et Vectorize (vtracer)** : repris tels quels depuis crispz-ideogram,
  déjà testés (`test_vector`, `test_local_tools`) ; rien à changer.
- **Edit in crispz-flux3** : repris ; le texte de l'onglet ne dit plus « MOSS-Image
  n'édite pas » mais « édition multi-référence NATIVE » (MOSS édite par la LoRA du lot
  4b). `cz_handoff` généralisé à deux cibles (`TARGETS` : dossier, env, clé de config),
  même garde « l'instance qui répond est cette app, pas la cible » (`_route`).
- **Compare in crispz-ideogram** (nouveau, onglet Input Image) : le prompt principal est
  rendu par crispz-ideogram à la même taille, même seed et mêmes steps, par son CLI du
  protocole v1 (`gen --spec -`, instance si son app tourne, sinon `--local` après
  libération de notre VRAM). Un layout MOSS est traduit en légende Ideogram 4 par
  `cz_caption.to_ideogram` : `high_level_description` = résumé, `background` = résumé
  (Ideogram l'exige), éléments `text` (texte verbatim, `desc` = caption ou « text ») et
  `obj` (`character: ...` pour un personnage), boîtes reprises telles quelles (même
  grille 0..1000 yxyx), palettes <= 5 par élément et <= 16 globale, `style_description`
  seulement si meta ou palette (aesthetics = meta, défauts « even lighting »,
  `graphic_design`, « flat graphic design », dits dans les notes), ordre des clés du
  schéma, JSON minifié UTF-8. Texte libre : tel quel. Le prompt envoyé est affiché dans
  l'onglet et gardé dans `handoff/<date>/ideogram_prompt.json` (ou `.txt`),
  `handoff.json` porte les notes et la licence. **Licence Ideogram 4 non commerciale**
  dite dans l'onglet, le log (`HANDOFF:`) et le fichier : une comparaison, jamais un
  livrable. Réglage `crispz_ideogram_dir` (env `CRISPZ_IDEOGRAM_DIR`, champ + Save dans
  l'onglet). Ports : dits dans l'onglet (deux apps sur 7860, la seconde prend 7861 ;
  `cli_protocol.instance_url` de crispz-ideogram à régler sur le port réel).
- Tests : `test_caption` (+1 : schéma Ideogram, ordre des clés, boîtes gardées, textes
  verbatim, hex majuscules, minifié, personnage, texte vide écarté, texte libre tel
  quel, layout compilé), `test_handoff` (+2 : faux crispz-ideogram en sous-processus,
  instance / froid / collision de port, spec gen avec taille, seed et steps, fichiers,
  licence loggée ; réglages séparés de ceux de crispz-flux3), `test_local_tools`
  (l'onglet nomme Ideogram 4 : admis par le test de vocabulaire).
- Vérifié : l'UI se construit avec le nouvel onglet (test) ; pas d'essai Chromium.
- À essayer chez Mike : `crispz_ideogram_dir` pointé sur son install, app crispz-ideogram
  ouverte puis fermée, un preset (Logo) rendu des deux côtés au même seed.
- **Lot 5 VALIDÉ par Mike le 05/11/2026.**

### Lot 6 (05/11/2026) : tests, doc, livraison, livré sur `wip/moss-engine`

Lancé sur « Lot 5 validé, passe au step suivant » de Mike.
- **Chaîne des licences vérifiée à la source** (fiches HF et fichiers LICENSE, GitHub
  OpenMOSS, lus le 05/11/2026) : poids MOSS-Image SFT et Base **Apache 2.0** (badge, texte
  de la fiche « weights and bundled inference code », LICENSE complet) ; code du pipeline
  Apache 2.0 (GitHub) ; Qwen3.5-9B Apache 2.0 (LICENSE « Copyright 2026 Alibaba Cloud ») ;
  FLUX.2-klein-4B Apache 2.0 au niveau du repo (« fully open under Apache 2.0 »), aucun
  fichier de licence propre au dossier `vae/` ; la fiche SFT dit « the bundled VAE uses
  the FLUX.2 small decoder » et ne nomme pas l'encodeur (identifié par sa config
  `Qwen3_5Model` et les noms de fichiers Comfy-Org). Comfy-Org, DeBand, LoRA DMD : badges
  Apache 2.0. **Moss-Image-Edit : aucune licence pour les poids** (code GPLv3). Le § 2 et
  le § 6 du cahier sont donc confirmés ; le « à confirmer » du VAE est levé.
  Point d'attention noté : la fiche klein 4B cite aussi des « Self-Hosted Commercial
  License Terms » et la licence non commerciale, qui visent le 9B ; le 4B reste Apache.
- **`tests/test_moss_e2e.py`** : chargement, 1024 avec le layout du preset Logo, 1456x1456
  en texte libre (`CZ_E2E_2MPX=0` pour sauter), img2img 0,35 (`_refine_whole`), inpaint
  (`inpaint_run`, hors masque identique au pixel), Stop au step 3 ; sauté sans CUDA ou
  sans accès au repo ; `tests/e2e_timings.txt`, images `tests/e2e_*.png` (gitignorées).
  Passé sur la 5090 le 05/11/2026, voir plus bas.
- **README** : bannière 0.1.0-dev « les six lots sont livrés », tableau de la chaîne des
  licences avec ce que chacune permet, mesures de tous les lots (img2img, LoRAs, édition),
  section Tests avec le e2e, section Pinokio. **FORK.md** : tableau des différences
  complété (LoRAs, édition, passerelles, catalogue, offload), mesures en bref.
- **Launcher Pinokio** préparé dans `pinokio/` (repris de celui de crispz-ideogram,
  `66b70a6` de son historique) : clone de crispz-moss, venv 3.11, mêmes règles que
  `install.bat` (Pillow 13 à part, extras après), torch 2.11 cu128, vérification
  `Qwen3_5Model` + `AutoencoderKLFlux2` + outlines + vtracer, message de fin sans token ni
  licence à accepter (27 Go au premier rendu), README avec la chaîne des licences et les
  mesures. Scripts vérifiés par `node` (ils se chargent). **Icône provisoire** (un « M »
  sur fond vert, dessiné avec PIL) : à remplacer par un logo généré avec l'app (preset
  Logo), enfin commercialisable. Comme pour Ideogram, le dossier quittera ce repo quand
  Mike aura créé `crispz-moss.pinokio` ; il clone `main`, donc utile après la fusion.
- Suite CPU : 70 fichiers (le e2e sauté sans CUDA) ; smoke 22/22.
- **Reste à décider par Mike** : fusion de `wip/moss-engine` dans
  `main`, passage d'`APP_VERSION` de 0.1.0-dev à 0.1.0, création du repo
  `crispz-moss.pinokio` (et son icône), le preset « Draft (8 steps, DMD LoRA) », le
  verdict nf4 contre bf16 sur images, et les rendus d'outpaint / reframe du passage de
  09:25 toujours non reçus.

### e2e GPU du lot 6 sur la 5090 (05/11/2026, Mike) : PASSÉ

`.venv\Scripts\python -m pytest tests\test_moss_e2e.py -s` (pytest 9.2.0 installé seul
dans le venv : il n'est pas dans `requirements.txt`, le CI l'ajoute à part), offload
'auto' -> 'model' encodeur seul, encodeur bf16, swap du DiT actif, 50 steps, CFG 3,0,
modèle en cache. **1 passed en 4 min 56**, les six étapes avec leurs assertions
(inpaint hors masque identique au pixel, Stop qui rend l'image partielle).

| étape | temps | pic réservé | référence des lots 1 à 3 |
|---|---|---|---|
| chargement depuis le cache | 34,9 s | 9,73 Go au repos | 28,5 s |
| 1024, layout du preset Logo (premier rendu : prompt et négatif neufs) | 83,0 s (débruitage 48 s, 1,03 it/s) | 20,47 Go | 85 à 90 s au lot 2, 42,7 s à chaud |
| 1456x1456 en texte libre (prompt neuf) | 108,7 s (2,02 s/it) | 20,44 Go | 101,7 s |
| img2img 0,35 (18 steps sur 50, prompt en cache) | 15,2 s | 15,83 Go | 1,2 it/s, cohérent |
| inpaint 1024 (50 steps, prompt neuf) | 47,4 s | 19,06 Go | 43,4 s à 1,9 Mpx plafonné |
| Stop au step 3 | 2,8 s | 15,33 Go | 0,8 s au lot 1 (step 19) |

Lecture :
- Les temps sont ceux des lots 1 à 3 : la 5090 était libre, rien à expliquer (le e2e
  d'Ideogram avait tourné 1,8 fois plus lent sur une carte partagée).
- **Le pic sur un prompt neuf dépend de la taille du prompt et de l'image** : 19,1 Go à
  1024 avec le prompt court de l'inpaint (ligne `vram: after the text: 19.1 GB`), 20,5 Go
  avec le layout Logo, 23,5 Go au lot 3b avec le long layout de l'affiche à 1,9 Mpx. Le
  « 17 à 19 Go » espéré au lot 3b était donc juste pour 1024 et faux pour l'affiche : ce
  sont les temporaires de l'encodeur (noyaux GDN en repli torch, float32) qui grandissent
  avec la longueur du prompt. Le DiT est bien garé pendant l'encodage (sinon 27 à 30 Go).
- Sur un prompt en cache, l'encodeur ne revient pas : `after the text: 9.8 GB` pour
  l'img2img, pic 15,8 Go.
- Suite au premier rendu, 35 s hors débruitage : deux encodages (négatif + prompt) avec
  le swap du DiT et les noyaux CUDA à froid. C'est le coût connu du premier rendu du
  processus, pas une régression.
- Les images `tests/e2e_*.png` restent à regarder (textes du logo, zone inpaintée), le
  test ne juge que les assertions.

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
