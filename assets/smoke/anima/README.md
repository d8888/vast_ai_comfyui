# smoke test 素材

本目錄存放 smoke test 需要上傳到 ComfyUI `input/` 的小型測試圖片，全部由
`tools/make_smoke_assets.py` 以純標準函式庫（不使用 PIL）確定性產生。

## `vast_comfy_smoke_control.png`

- RGB、1024×1024、白底黑色細線稿（貓臉與場景，數十筆筆畫，無隨機成分）。
- 用途：案例 2「Base ＋ LLLite 混合線稿控制」的控制圖，交給
  `anima-lllite-any-test-like-v2.safetensors`。
- 高對比、乾淨的線稿，適合作為 any-test-like 模型的控制輸入（不是雜訊圖）。

## `vast_comfy_smoke_inpaint_src.png`

- RGBA、1024×1024、彩色簡單風景（漸層天空、太陽、山丘、房子、樹）。
- 中央有一個完全透明的圓形區域（alpha = 0），代表要重繪的區域；
  ComfyUI `LoadImage` 的 MASK 輸出為 `1 - alpha`，因此該圓形即 mask = 1
  （重繪區）、其餘為 mask = 0（保留區）。
- 透明區域下方的 RGB 仍保留合理內容（房子），符合局部重繪輸入的預期。
- 用途：案例 3「Base ＋ LLLite 局部重繪」（`anima-lllite-inpainting-v2.safetensors`），
  LoadImage 的 IMAGE（slot 0）與 MASK（slot 1）同時接到外掛節點。

## `smoke_cases.yaml`

smoke runner 的案例參數檔（案例順序、逾時、需上傳的 `inputs`、節點參數
`overrides` 與 `expect` 尺寸）。KSampler（節點 `"6"`）的 seed 使用
`"$run_seed"` 佔位，由 runner 在每次執行時替換成新的整數 seed。

## 重新產生素材

在本 repository 根目錄執行：

```text
python tools/make_smoke_assets.py
```

或指定輸出目錄：

```text
python tools/make_smoke_assets.py --out-dir assets/smoke
```

特性：無隨機、可重現（同樣輸入必得同樣位元內容）、每個檔案遠小於 300 KB
（PNG filter type 0、zlib level 9）。腳本結束時會列出每個檔案的路徑與大小。
