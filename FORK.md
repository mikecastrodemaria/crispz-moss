# crispz-moss : fork MOSS-Image de crispz-ideogram

- Base : `main` de crispz-ideogram (4d028a3, 0.1.0), avec ses 446 commits.
- Remotes : `origin` = crispz-moss, `ideogram` = crispz-ideogram (pour reprendre les
  correctifs communs).
- Pourquoi cette base et pas crispz-flux3 : l'éditeur de légende avec boîtes à la souris,
  les presets de design, le magic prompt et le Describe, l'img2img et l'inpaint émulés
  sur ce même VAE, l'offload « encodeur seul » déjà écrit pour un encodeur Qwen de 8 à
  9B. Le détail est dans `CLAUDE.md` §3.

## Ce qui change par rapport à crispz-ideogram

| | crispz-ideogram | crispz-moss |
|---|---|---|
| Modèle | Ideogram 4.0 (2 x 9,3B, nf4, licence non commerciale, repo gated) | MOSS-Image (5B, bf16, Apache 2.0, public) |
| Moteur | `cz_ideogram.py` sur `Ideogram4Pipeline` de diffusers | `cz_moss.py` sur le pipeline OpenMOSS vendorisé (`third_party/moss_image`) |
| Encodeur | Qwen3-VL-8B, 13 couches cachées | Qwen3.5-9B, dernière couche, bf16 ou nf4 |
| Prompt négatif | aucun (branche inconditionnelle sans texte) | réel |
| Guidance | planning par step (pic 7,0) | CFG constante (3,0) |
| Légende | JSON Ideogram (boîtes en millièmes, vérificateur officiel) | layout MOSS (boîtes en pixels, compilé en coordonnées centrées) |
| Fichiers uniques | refusés (pas de convertisseur) | ComfyUI / CivitAI acceptés (bf16, fp8, int8 ConvRot), 4 bits refusés |
| Cloud | fonctions Ideogram 4.5 (cachées) | aucune |
| img2img / inpaint | émulés | émulés (même technique, ordre de tokens FLUX) ; outpaint et reframe avec le draft placé et les bandes neuves déclarées |
| LoRAs | mixin diffusers (`Ideogram4LoraLoaderMixin`) | chargeur maison par hooks (kohya, peft, moss-edit-lora, diffs ComfyUI) |
| Édition par références | aucune (cloud Ideogram 4.5 caché) | LoRA Moss-Image-Edit, préfixe de références sur le DiT vendorisé (`cz_moss.EditPrefix`) |
| Passerelles | Edit in crispz-flux3 | Edit in crispz-flux3, Compare in crispz-ideogram (layout traduit en légende Ideogram 4) |
| Catalogue | aucun | MOSS-Image Character Catalog dans l'Asset Browser, « Send to crispz-moss » |
| Offload | encodeur seul | encodeur seul, le DiT garé pendant l'encodage (`moss_encode_swap`), texte encodé avant tout VAE |

Les notes de crispz-ideogram sont gardées dans `FORK-ideogram.md` et
`FORK-ideogram-CLAUDE.md` (pièges d'environnement, mesures), celles de crispz-flux3 dans
`FORK-flux3.md` et `FORK-flux3-CLAUDE.md`.

## Mesures

Voir `README.md` (tableau RTX 5090) et `CLAUDE.md` §9 pour les passages détaillés. En
bref, sur la même carte : 1024 en 43 s contre 61 s pour Ideogram 4, 2 Mpx en 102 s contre
394 s pour 4 Mpx, pic 15 Go contre 23 ; A/B texte 24 rendus sur 24 exacts. Le prix : 27 Go
de poids bf16 (un encodeur de 16,8 Go en RAM entre deux prompts) contre 15 Go nf4.
