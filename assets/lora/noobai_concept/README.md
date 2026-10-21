# LoRA track

A concept LoRA trained from a handful of generated images on the instance's own
base checkpoint. The point of this track is **reproducibility**, not image
quality: any run must be repeatable from the committed material plus the pinned
training stack.

## Layout

| path | role |
|---|---|
| `config/lora/<name>.yaml` | track config: base checkpoint, steps, dim/alpha, tag threshold |
| `assets/lora/<name>/train/` | the training images **and** their WD14 captions |
| `assets/lora/<name>/verify/` | LoRA-off / LoRA-on pair proving the weights do something |
| `assets/lora/<name>/wd14_report.json` | tagger output per image (the raw evidence behind the captions) |
| `src/vast_comfy/lora.py` | plan builder, version pins, Hugging Face publisher |

## Pinned stack

A run is only reproducible when the toolchain is exact. `lora.pins()` holds
these and `tests/test_lora.py` asserts none of them is a floating version:

| component | pin |
|---|---|
| kohya sd-scripts | `11d8cedd0c67b93a10f8b2d75f43554f6f12c794` |
| torch | `2.10.0+cu130` (+ torchvision `0.25.0+cu130`) |
| transformers | `5.17.0` |
| onnxruntime (tagger) | `1.31.0` |
| WD14 tagger | `SmilingWolf/wd-v1-4-convnextv2-tagger-v2` @ `dbd4dbe553ee51feb3bc745b614fb762080e3912` |

Training venv is separate from ComfyUI's: `uv venv` + the pinned torch index.
ComfyUI's own venv is never modified.

## Pipeline order

```
generate 5 images (same fixed concept, one varied view clause each)
  -> WD14 tag  (Danbooru tags = NoobAI's native vocabulary)
  -> sdxl_train_network.py  (LoRA, dim 16 / alpha 8, AdamW8bit, fp16, SDPA)
  -> load into ComfyUI LoraLoader
  -> verify: same seed + prompt, LoRA off then on, measure the difference
```

Order matters and `lora.build_plan()` encodes it. Verification is part of the
plan because a `.safetensors` that exists but produces the same image as the
base model is a **failed** run.

## Verified smoke run (2026-10-21)

| item | value |
|---|---|
| instance | RTX 3090 24 GB, ComfyUI 0.41.0, torch 2.10.0+cu130 |
| images | 5 × 1024×1024, concept `grey-purple short hair, white stand collar jacket, black pants` |
| tagging | WD14 v1.4 convnextv2, threshold 0.35 — 17–30 general tags per image, 0 characters |
| training | 500 steps, 10 epochs, 7 m 29 s, final avr_loss ≈ 0.064 |
| weights | `lora_smoke_concept.safetensors`, 114 431 572 B, 2958 tensors, 57 009 114 params |
| verification | mean abs diff **16.06** / 255, **61.43 %** of pixels changed by > 8 |

The verification pair in `verify/` is the actual evidence: same seed, same
prompt, only the LoraLoader differs. The LoRA-on image pulls the jacket toward
an over-shirt silhouette and lightens the hair — the network learned garment and
hair structure, which is what the concept was supposed to carry. Quality is
mediocre, and that is expected from 5 images / 500 steps.

## Known limitations found in this run

- **SDXL has no `clip_skip`.** `--clip_skip=2` is silently ignored by
  `sdxl_train_network.py`; passing it only produces a warning. Do not carry the
  SD1.5 habit over.
- **`--cache_text_encoder_outputs` is incompatible with training the text
  encoder.** It asserts out. Either drop the flag or add
  `--network_train_unet_only`. This run dropped the flag.
- **5 images is below the point where quality is meaningful.** Underfitting is
  normal; treat any success claim here as "the chain works", never "the LoRA is
  good".
- The generated set has one shared weakness: hands and sleeves are the first
  things to smear. For a real LoRA, curate out bad hands before training.

## Hugging Face push

`lora.HfPublisher` uploads the weights file via
`PUT https://huggingface.co/api/models/<repo>/upload/<path>`.

- Token comes from `HF_TOKEN` **at call time** and is never stored on the
  object, so it cannot leak through a repr or a log line.
- `dry_run=True` needs no token and sends nothing.
- Missing token or a non-2xx response fails closed with `E_HF_PUSH_FAILED`.
- The whole path is tested with a mock transport (`tests/test_lora.py`); no
  network and no real token are needed in CI.

Set `hf_repo: "<user>/<repo>"` in the track config to enable it for a real run.

## Re-running

```bash
# on the instance
uv venv /workspace/sd-scripts/.venv --python 3.12
uv pip install --python /workspace/sd-scripts/.venv/bin/python \
    torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /workspace/sd-scripts/.venv/bin/python -r requirements.txt
```

Then tag with the pinned WD14 revision, train with the parameters from
`config/lora/noobai_concept.yaml`, and verify with the same seed/prompt pair.

## Files in this directory

| file | what it is |
|---|---|
| `train/lora_ds_*.png` | the 5 training images, all 1024×1024 |
| `train/lora_ds_*.txt` | WD14 captions, one space-separated tag line per image |
| `verify/lora_verify_without_00001_.png` | seed 4242424242, LoRA off |
| `verify/lora_verify_with_00001_.png` | seed 4242424242, LoRA on, strength 1.0 |
| `wd14_report.json` | per-image tagger scores, the evidence behind the captions |

Prompt used for every training image (only the trailing view clause varies):

```
masterpiece, best quality, newest, absurdres, highres, safe, 1girl, solo,
grey-purple short hair, white stand collar jacket, black pants,
white background, simple background, looking at viewer
```

Views: `full body, from front, standing, arms at sides` /
`from side, three-quarter view` / `from behind` / `upper body, portrait` /
`sitting, from front`.

The weights file itself is **not** committed (114 MB). Reproduce it with the
pinned stack above, or push it to Hugging Face with `lora.HfPublisher`.
