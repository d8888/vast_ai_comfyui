# vast-comfy 工作流

本目錄提供 smoke test 所需的四套工作流，皆使用 Anima 模型家族
（`anima-base-v1.0.safetensors`、`anima-turbo-v1.1.safetensors`）、
`qwen_3_06b_base.safetensors` 文字編碼器與 `qwen_image_vae.safetensors` VAE。

## 四個測試案例（依 SPEC.md 13.1 的順序）

| 順序 | 檔案（API） | 內容 |
| --- | --- | --- |
| 1 | `api/anima_base_t2i.api.json` | Anima-Base 文生圖，1024×1024、30 steps、CFG 4 |
| 2 | `api/anima_base_lllite_lineart.api.json` | Base ＋ LLLite 混合線稿控制（`anima-lllite-any-test-like-v2.safetensors`），1024×1024、30 steps、CFG 4 |
| 3 | `api/anima_base_lllite_inpaint.api.json` | Base ＋ LLLite 局部重繪（`anima-lllite-inpainting-v2.safetensors`），使用圖片＋mask，1024×1024、30 steps、CFG 4 |
| 4 | `api/anima_turbo_t2i.api.json` | Anima-Turbo 文生圖，1024×1024、8 steps、CFG 1 |

四個案例共用同一組安全測試提示詞（見工作流內容），且 KSampler 的 seed 由
smoke runner 依 `assets/smoke/smoke_cases.yaml` 的 `overrides`（節點 `"6"`）
在每次執行時以新的整數覆寫，以避免 ComfyUI 快取導致「重用舊結果」。

## API 格式（`workflows/api/*.api.json`）

- 為 ComfyUI「Save (API Format)」格式，也就是 `POST /prompt` 的 payload 中
  `prompt` 欄位的內容：`{"<node id>": {"class_type": ..., "inputs": {...}, "_meta": {...}}}`。
- smoke runner 直接讀取檔案後送出執行；節點 id `"1"`～`"11"` 固定，其中
  `"6"` 一律是 KSampler。
- LLLite 兩個案例使用外掛節點 `AnimaLLLiteApply_sdscripts`
  （kohya-ss/ComfyUI-Anima-LLLite），不是 ComfyUI 內建的 `AnimaLLLiteApply`。

## GUI 格式（`workflows/gui/*.json`）

- 為 ComfyUI 網頁介面可開啟的 Litegraph 格式，內容與 API 工作流相同，
  供使用者手動檢視／微調與比對。
- 開啟後請確認 LoadImage 使用 `assets/smoke/` 內的上傳檔名
  （`vast_comfy_smoke_control.png`、`vast_comfy_smoke_inpaint_src.png`）。

## 前置需求

- 模型：Anima-Base v1.0、Anima-Turbo v1.1（完整 checkpoint，非 LoRA）、
  `qwen_3_06b_base.safetensors`、`qwen_image_vae.safetensors`。
- 外掛：`ComfyUI-Anima-LLLite`（固定 commit），其權重依外掛契約放在
  **`models/controlnet/`**：`anima-lllite-any-test-like-v2.safetensors`、
  `anima-lllite-inpainting-v2.safetensors`。不可與內建節點使用的
  `models/model_patches/` 混用。
