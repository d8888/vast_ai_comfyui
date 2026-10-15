# vast-comfy

以硬性條件在 Vast.ai 上找到**報價最低**的合格 GPU instance，明確授權後租用，部署官方 ComfyUI image、Anima 模型家族與 Anima-LLLite 外掛，並以**實際出圖**驗收可用性。

- 完整已批准規格： [`SPEC.md`](./SPEC.md)
- 本 README 對應規格版本 1.0。**目前尚未做過真機付費驗收**（見文末「驗證狀態」）。

---

## 目錄

1. [快速開始](#1-快速開始)
2. [安裝與前置條件](#2-安裝與前置條件)
3. [秘密值與 SSH 金鑰](#3-秘密值與-ssh-金鑰)
4. [設定檔、GPU alias 與覆寫順序](#4-設定檔gpu-alias-與覆寫順序)
5. [搜尋、報價與付費租用](#5-搜尋報價與付費租用)
6. [兩個完整組合流程](#6-兩個完整組合流程)
7. [連線：Proxy SSH、SSH tunnel 與官方 Portal](#7-連線proxy-sshssh-tunnel-與官方-portal)
8. [獨占 session、心跳與衝突處置](#8-獨占-session心跳與衝突處置)
9. [背景 job：狀態、日誌、等待與錯誤接手](#9-背景-job狀態日誌等待與錯誤接手)
10. [模型、外掛與固定版本](#10-模型外掛與固定版本)
11. [Smoke test 與 READY 判定](#11-smoke-test-與-ready-判定)
12. [下載圖片／報告／outputs](#12-下載圖片報告outputs)
13. [檔案傳輸邊界與安全](#13-檔案傳輸邊界與安全)
14. [費用、資料保存與 stop／destroy](#14-費用資料保存與-stopdestroy)
15. [命令與 JSON 契約](#15-命令與-json-契約)
16. [常見問題排解](#16-常見問題排解)
17. [模型授權與驗收範圍](#17-模型授權與驗收範圍)
18. [驗證狀態](#18-驗證狀態)

---

## 1. 快速開始

```bash
# 1) 安裝（見下一節的完整說明）
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux
pip install -e ".[dev]"

# 2) 設定 API key（.env 不進版控）
copy .env.template .env           # Windows（Linux: cp）
# 編輯 .env，填入 VAST_API_KEY

# 3) 註冊 SSH 公鑰（必須在租用前完成）
vast-comfy keys generate --path ~/.ssh/id_ed25519   # 若已有金鑰則跳過
vast-comfy keys register --pubkey ~/.ssh/id_ed25519.pub

# 4) 看報價（不付費）
vast-comfy --env-file .env offers best --alias rtx_4060_ti_16gb

# 5) 租用＋部署＋驗收（付費；--confirm-paid 是必要旗標）
vast-comfy --env-file .env flow create-and-prepare \
  --alias rtx_4060_ti_16gb --confirm-paid

# 6) 打開 ComfyUI（本機 127.0.0.1:8188）
vast-comfy --env-file .env ssh tunnel --instance <INSTANCE_ID> --start
```

之後：
- 瀏覽器開 <http://127.0.0.1:8188/>（SSH tunnel，直達 ComfyUI）。
- 或到 Vast.ai 管理頁面按實例卡片的 **Open**（官方帶驗證 Portal）。

> `--env-file .env` 必須明確指定；程式不會自動讀取 `.env`。

---

## 2. 安裝與前置條件

| 項目 | 需求 |
| --- | --- |
| 控制端 OS | Windows 10/11 或 Linux |
| Python | 3.10 以上（開發驗證使用 3.14） |
| OpenSSH client | Windows 內建；Linux 安裝 `openssh-client`。需有 `ssh` 與 `scp` 指令 |
| Python 套件 | `PyYAML`、`requests`（`pip install -e .` 會自動安裝） |

```bash
python -m venv .venv
.venv\Scripts\activate               # Windows
source .venv/bin/activate            # Linux
pip install -e ".[dev]"              # dev 額外包含 pytest / pillow
vast-comfy --version
vast-comfy config validate           # 檢查設定、alias、模型清單、smoke 案例
python -m pytest                     # 測試套件（詳見 tests/README.md）
```

**Agent 用法提醒**：所有命令加 `--json` 會輸出單一 JSON 物件（日誌全部走 stderr），失敗有穩定錯誤碼與非零 exit code；請見第 15 節。

不使用虛擬環境也可以，只要 `python -m vast_comfy ...` 能匯入套件即可。

---

## 3. 秘密值與 SSH 金鑰

### API key / token

- 所有秘密值從**環境變數**讀取；可明確 `--env-file .env` 載入 `.env`（覆蓋既有環境變數）。
- YAML 只會出現「環境變數名稱」，不會出現秘密值。
- `.env` 已被 `.gitignore` 排除；**不要**把任何秘密值寫進 `config/*.yaml`。

`.env.template` 內容（複製為 `.env` 後填入）：

| 變數 | 用途 |
| --- | --- |
| `VAST_API_KEY` | 必要。Vast.ai API key（<https://cloud.vast.ai/manage-keys/>），只留在控制端 |
| `HF_TOKEN` | 選用。只有加入 gated/私有 Hugging Face 檔案時才需要 |
| `CIVITAI_TOKEN` | 選用。只有加入 CivitAI 檔案時才需要 |

Vast.ai API key **永遠不會**上傳到 instance；`HF_TOKEN`／`CIVITAI_TOKEN` 也只有在你明確批准後才會用於 instance 端下載（見第 13 節）。

### SSH 金鑰

- v1 只支援 **OpenSSH 格式**（`id_ed25519` / `id_rsa`）。**不支援 PuTTY／PPK**；若你只有 PPK，請自行用 PuTTYgen 轉出 OpenSSH 金鑰，本工具不做轉換。
- 使用既有金鑰：在 YAML 設定 `ssh.key_path`，例如 `~/.ssh/id_ed25519`，或使用 `ssh-agent`（不設 key_path 時走 agent）。
- 建立新金鑰（**不覆寫既有檔案**）：

```bash
vast-comfy keys generate --path ~/.ssh/id_ed25519   # 已存在時會直接失敗（E_KEY_EXISTS）
vast-comfy keys register --pubkey ~/.ssh/id_ed25519.pub
vast-comfy keys list
```

- **公鑰必須在租用前完成註冊**：帳號金鑰只會注入「之後建立」的 instance。
- 私鑰永遠留在控制端；已知主機檔案是獨立的 `~/.vast_comfy/known_hosts`，首次連線採 TOFU（`accept-new`），之後主機指紋改變會直接失敗，**不會**自動停用主機驗證。

---

## 4. 設定檔、GPU alias 與覆寫順序

### 優先順序

**內建預設 → YAML（`--config`）→ 明確 CLI 參數（`--set PATH=VALUE`）**

- 未知欄位、重複 YAML key、型別／範圍／矛盾設定、未知 alias 一律在任何副作用前失敗。
- `config/default.yaml` 是「內建預設的參考副本」，由測試確保同步。全部預設值見該檔。

常用覆寫範例：

```bash
vast-comfy --set storage.disk_gb=500 --set pricing.max_dph_total_usd_per_hour=1.5 offers search --alias rtx_4090
vast-comfy --config my-config.yaml flow prepare --instance 123456
```

### GPU alias

```bash
vast-comfy gpu aliases                 # 列出全部（含自訂）
vast-comfy gpu aliases --custom-only
vast-comfy gpu resolve --alias rtx_4090
vast-comfy gpu update-aliases          # 明確更新官方型號目錄（保留自訂 alias，永不刪除型號）
vast-comfy gpu add-alias --alias my_4090 --gpu-name "RTX 4090"
```

- alias 必須精确命中；未知 alias 直接失敗（可提示相近名稱，但不會自動修正）。
- 官方型號目錄快照在 `config/gpu_aliases.json`；`RTX 4060 Ti` 這類型號的官方名稱不區分 VRAM，所以提供帶條件的 alias，例如 `rtx_4060_ti_8gb` 與 `rtx_4060_ti_16gb`（附 `gpu_ram` 篩選）。
- 你的自訂 alias 存放於同一檔的 `custom` 區塊；`gpu update-aliases` 只會重建官方區塊。

### 硬性條件（預設值）

| 項目 | 預設 |
| --- | --- |
| GPU | 由 alias 指定；**必填**，無預設型號 |
| GPU 數量 | 恰好 1 張（精確匹配，非「至少」） |
| 硬碟 | 300GB（搜尋報價時就用此容量計算整合價） |
| 平台回報網速 | 上下傳各 ≥ 500（與官方 Vast.ai ComfyUI 模板相同的比較方式） |
| 租約 | 一律 on-demand |
| 時租上限 | USD $2/hour（以整合欄位 `dph_total` 比較；流量費另計） |
| 主機 | verified、reliability ≥ 0.99 |
| CPU / RAM | ≥ 4 effective vCPU、≥ 16GB 系統 RAM |
| 架構 | NVIDIA CUDA、x86-64 |
| Image 相容性 | `cuda_max_good ≥ 12.9`、`compute_cap ≥ 750`（對應固定的 CUDA 12.9 image） |
| 地區 | 預設不限；可用 `host.geolocation_allow` / `host.geolocation_exclude` |
| SSH | `proxy_only`（v1 唯一策略） |

全部都是硬性條件：**沒有任何一項會被私下放寬**。無合格 offer 或超預算都會失敗，不會換 GPU、不會加價、不會自動改租。

---

## 5. 搜尋、報價與付費租用

```bash
# 搜尋（唯讀、不付費）：列出合格候選與最低價
vast-comfy --env-file .env offers search --alias rtx_4060_ti_16gb --top 10

# 只要最低價那筆
vast-comfy --env-file .env offers best --alias rtx_4060_ti_16gb

# 付費建立（務必先確認報價；--confirm-paid 為必要旗標）
vast-comfy --env-file .env rent create \
  --alias rtx_4060_ti_16gb --offer-id 123456789 --label my-run --confirm-paid --wait

# 若建立結果不明（網路中斷等）：先查證，不要重送
vast-comfy --env-file .env rent verify --label my-run
```

規則：

- 報價用**套用指定硬碟容量後的整合時租** `dph_total` 排序與比對上限；下載／上傳流量單價會單獨列出（`inet_up_cost_usd_per_gb` 等）。
- 建立前會重新查一次該 offer：條件改變、價格超上限或被租走 → `E_OFFER_CHANGED` / `E_PRICE_LIMIT`，不會自動改租下一台。
- 建立請求**永不自動重試**；結果不明回傳 `E_CREATE_UNCERTAIN`，附查證方式。
- 不使用 reserved／interruptible、不自動加值、不提高上限。
- 時租上限**不是**累積總花費上限；流量費不受上限約束。

---

## 6. 兩個完整組合流程

### A. 準備既有 instance

```bash
vast-comfy --env-file .env flow prepare --instance <ID>
```

遠端依序執行：取得獨占 session → 安裝模型／外掛 → 必要時重啟 ComfyUI（**佇列忙碌時絕不重啟**）→ GPU／節點檢查 → 完整 smoke suite → 寫入遠端報告。

- 這是**背景 job**：本機斷線不會中止遠端流程。
- 成功後預設釋放獨占權（`session.release_on_success: true`，或使用 `--keep-session` 保留）。
- 失敗時：明確停止後續階段、保留遠端現況、**不自動清理**，錯誤訊息會說明鎖與計費狀態。

### B. 建立並準備新 instance

```bash
vast-comfy --env-file .env flow create-and-prepare \
  --alias rtx_4060_ti_16gb --confirm-paid
```

流程：驗證設定／金鑰 → 搜尋與報價 → `--confirm-paid` → 建立並查證 → 有上限的啟動等待 → SSH 實際登入驗證 → 準備流程（同 A）。

任一步無法安全確定時就明確失敗；不會偷偷換機器、換 image 或改條件。

---

## 7. 連線：Proxy SSH、SSH tunnel 與官方 Portal

### Proxy SSH（v1 唯一策略）

```bash
vast-comfy --env-file .env ssh info --instance <ID>   # 連線資訊 + tunnel 指令 + Portal 資訊
vast-comfy --env-file .env ssh test --instance <ID>   # 實際登入並執行命令才算通過
```

- 使用 instance 的 `ssh_host:ssh_port`（Vast.ai proxy 位址）；直接連線埠即使存在也會被刻意忽略。
- SSH 就緒標準是「實際登入且命令成功」，不是 TCP 埠可連。

### SSH tunnel（主要存取方式）

```bash
vast-comfy --env-file .env ssh tunnel --instance <ID> --print   # 只印指令
vast-comfy --env-file .env ssh tunnel --instance <ID> --start   # 背景啟動，綁 127.0.0.1:8188
vast-comfy --env-file .env ssh tunnel --instance <ID> --stop
```

- 本機端只綁定 `127.0.0.1`，不暴露到區域網路。
- 瀏覽器開 <http://127.0.0.1:8188/>；tunnel 直達 ComfyUI 後端（內部埠 18188），本機不需再登入。
- tunnel 是控制端本機程序；關掉它不影響 instance 運作。

### 官方 Portal

- 到 Vast.ai 管理頁面按實例卡片的 **Open**：入口會自動帶驗證 cookie；Portal 中的 ComfyUI 連結（外部埠 8188）同樣受官方驗證保護。
- Portal 需要 `OPEN_BUTTON_TOKEN`（平台自動產生）。本工具**不會**把含 token 的 URL 寫進日誌或報告；需要 token 時以 README 或 `ssh info` 顯示的非秘密方式取得。
- 兩條路都存在：tunnel 更快，Portal 適合快速檢視。

---

## 8. 獨占 session、心跳與衝突處置

同一時間只允許一個工具包 session 操作一個 instance；同一個 session 內也以**操作鎖**確保同一時間只有一個修改型操作。

```bash
vast-comfy --env-file .env session acquire --instance <ID>       # 取得獨占（僅先上傳最小鎖元件）
vast-comfy --env-file .env session status  --instance <ID>       # 唯讀診斷
vast-comfy --env-file .env session release --instance <ID> --session-id <SID>
vast-comfy --env-file .env session recover --instance <ID> --accept-stale   # 明確的人工復原
```

機制：

- 真正的互斥由 instance 上的 **Linux `flock`** 提供，由一個背景 daemon 全程持有；`owner.json`（session、取得時間、最後心跳、控制端、daemon pid）只用於診斷。
- **兩層排他**：`session` 鎖（跨控制端獨占）＋每次操作的**操作鎖**（同一 session 也不能同時跑兩個部署／下載／工作）。操作鎖由實際執行工作的程序或其背景 job 繼承持有，控制端斷線或 session daemon 結束都不會提前解除。
- 取得 session 前只會上傳**最小鎖元件**（`vc_common.py`／`vc_session.py` 的內容雜湊版本）；完整管理套件只在你持有所有權後才部署，衝突的另一方不會改到正在使用的程式碼。
- **心跳自動更新**：每 30 秒；有效期 300 秒；額外緩衝 60 秒。時間以**instance 的時鐘**為準。
- 衝突時立即失敗（exit code 4）並顯示持有者資訊；**不排隊、不搶鎖、不殺 Agent、不取消工作**。
- 心跳過期只「回報」（時間、持有者、進行中工作、實際鎖狀態），**不會**自動接手。
- 遠端工作存活期間，`flock` 實際持有，不代表過期就能插入；工作逾時只標記 `overdue` 並持續等待，**不會**代你終止。
- 釋放前若該 session 仍有存活工作或操作進行中 → 拒絕釋放（`E_LOCK_JOBS_RUNNING`／`E_LOCK_CONFLICT`）。

**為什麼不能靠刪除鎖檔解鎖？**
`flock` 綁在檔案的 inode 上：刪掉鎖檔並重建會產生另一個 inode，舊持有者仍鎖著舊檔案，兩邊都以為自己拿到鎖。因此本工具的任何流程都不會刪除或重建鎖檔；人工復原請在 instance 上確認舊持有者與工作都已結束後，使用 `session recover --accept-stale`（明確表達「你已親自確認」）。

發現衝突時的正確做法：

1. `session status` 查看持有者與其進行中的工作。
2. 到**持有者的控制電腦**結束它的流程（終止本機 Agent **不代表**遠端背景工作已停止）。
3. 確認遠端沒有存活工作後，再 `session recover --accept-stale`。
4. 你決定不了就維持現狀；本工具不會替你處置。

---

## 9. 背景 job：狀態、日誌、等待與錯誤接手

長任務（安裝、smoke、準備流程）都以遠端背景 job 執行：

```bash
vast-comfy --env-file .env jobs start --instance <ID> --name my-job --timeout-minutes 30 -- bash -lc "echo hi"
vast-comfy --env-file .env jobs status  --instance <ID> --job-id <JOB>
vast-comfy --env-file .env jobs logs    --instance <ID> --job-id <JOB> --tail 200
vast-comfy --env-file .env jobs wait    --instance <ID> --job-id <JOB> --timeout-minutes 60
vast-comfy --env-file .env jobs list    --instance <ID>
```

- job 是獨立程序：本機關機、SSH 斷線都不會中止；重新登入後用 `jobs status` 追蹤。
- **等待逾時不會停止 job**：`E_JOB_TIMEOUT` 會明確說明「job 可能仍在執行」。
- 個別工具完成**不會**自動接續下一步；只有組合流程會照它定義的階段跑完。

### 不可恢復錯誤的處理原則

- 一律明確失敗並如實回報：錯誤碼、階段、instance／job ID、工作與計費狀態。
- **不**自動 stop／destroy、**不**取消遠端工作、**不**解鎖、**不**重新租機。
- 你接手後自行決定要用哪個明確命令處理（見第 14 節）。

---

## 10. 模型、外掛與固定版本

清單在 [`config/models.yaml`](./config/models.yaml)，全部固定 revision／commit 與 sha256：

| 檔案 | 大小 | 目的地（ComfyUI） |
| --- | --- | --- |
| `anima-base-v1.0.safetensors` | 3.9 GB | `models/diffusion_models/` |
| `anima-turbo-v1.1.safetensors` | 3.9 GB | `models/diffusion_models/` |
| `qwen_3_06b_base.safetensors` | 1.1 GB | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | 242 MB | `models/vae/` |
| `anima-lllite-any-test-like-v2.safetensors` | 16 MB | `models/controlnet/` |
| `anima-lllite-inpainting-v2.safetensors` | 63 MB | `models/controlnet/` |
| 外掛 `kohya-ss/ComfyUI-Anima-LLLite`（commit `b7495bd`）| — | `custom_nodes/ComfyUI-Anima-LLLite` |

安裝規則：

- 全部由 **instance 直接下載**（支援續傳）；未完成檔與正式檔分離，校驗通過才轉正。
- 目標檔已存在且校驗相符 → 略過；**不符 → 失敗，絕不覆寫**。
- 外掛固定在指定 commit；有本機修改或版本不符 → 失敗，**不做 reset／不自動更新**。
- 目的路徑限定在 `models/` 允許子目錄與 `custom_nodes/<name>`，拒絕任何越界路徑。

```bash
vast-comfy --env-file .env install run    --instance <ID> --session-id <SID>
vast-comfy --env-file .env install verify --instance <ID> --session-id <SID>
```

### 工作流

- `workflows/gui/*.json`：在 ComfyUI 網頁 UI 開啟（4 套）。
- `workflows/api/*.json`：API 格式，smoke runner 使用；你也可以直接 POST `/prompt`。
- 兩者圖形一致：Base 文生圖、Base＋LLLite 混合線稿、Base＋LLLite 局部重繪、Turbo 文生圖。
- LLLite 必須使用外掛節點 `AnimaLLLiteApply_sdscripts` 與 `models/controlnet/` 內的權重；ComfyUI 內建同名節點不算通過。

---

## 11. Smoke test 與 READY 判定

```bash
vast-comfy --env-file .env smoke run --instance <ID> --session-id <SID>
vast-comfy --env-file .env smoke report --instance <ID> --session-id <SID> --run-id <RUN>
```

四項必測案例（依序，全部必須**真正出新圖**）：

| # | 案例 | 參數 |
| --- | --- | --- |
| 1 | Anima-Base 文生圖 | 1024×1024、batch 1、30 steps、CFG 4 |
| 2 | Base＋LLLite 混合線稿 | 同上，使用內附控制圖 |
| 3 | Base＋LLLite 局部重繪 | 同上，使用內附圖片＋mask |
| 4 | Anima-Turbo 文生圖 | 1024×1024、8 steps、CFG 1 |

驗收細節：

- 每次執行使用**該次專屬的 seed**（防 ComfyUI 節點快取讓舊結果冒充新結果），實際 seed 記錄在報告中。
- 每張圖必須：本次執行後才產生（mtime）、尺寸正確、可解碼、不是單一顏色。
- GPU 必須是 CUDA 裝置、`nvidia-smi` 正常；必要節點（含外掛節點）必須存在。
- 全部通過才回傳 `READY`；只跑部分案例為 `PARTIAL`（永遠不算 READY）；任一失敗為 `FAILED` 且**後續案例不執行**，明確停止。
- 發現 ComfyUI 已有排隊／執行中的生成工作 → `E_COMFY_BUSY`：不清佇列、不插隊、不強制重啟。
- OOM 或逾時不會偷偷降解析度、減 steps 或換模型。

報告（JSON＋Markdown）留在 instance：`/workspace/.vast_comfy/runs/<run_id>/`；圖片留在 ComfyUI 的 output 目錄。**不會**自動下載。

### 功能驗收範圍

通過 smoke test 代表「當下這台機器、這組版本能實際出圖」。它**不**保證：美術品質、LLLite 控制效果符合你的主觀期待、或主機未來永不離線。

---

## 12. 下載圖片／報告／outputs

下載一律是明確命令，先驗證遠端 sha256，且**不覆寫**本機既有檔案（除非 `--overwrite`）：

```bash
vast-comfy --env-file .env download report  --instance <ID> --session-id <SID> --run-id <RUN> --dest .\out
vast-comfy --env-file .env download images  --instance <ID> --session-id <SID> --run-id <RUN> --dest .\out
vast-comfy --env-file .env download outputs --instance <ID> --session-id <SID> --dest .\out --pattern "*.png" --limit 50
```

- `outputs` 是「最近 N 張」的一次性批次下載，不做持續同步。
- 本機檔案衝突 → `E_DOWNLOAD` 並列出衝突檔案；加了 `--overwrite` 才會取代。
- 不接 S3／Google Drive、不自動備份。

---

## 13. 檔案傳輸邊界與安全

### 允許自動上傳（白名單，內容雜湊定址）

- 本專案自己的管理程式：`src/vast_comfy/remote_agent/*.py`
- 本專案工作流：`workflows/api/*.json`、`workflows/gui/*.json`
- 小型非機密測試素材：`assets/smoke/*`（PNG／YAML／README）

封裝採明確白名單＋sha256 manifest；部署到 `/workspace/.vast_comfy/pkg/<hash>/`，遠端驗證通過後才切換 `current`。`.env`、私鑰、`known_hosts`、`*.pem` 等一律拒絕打包（有測試覆蓋）。

### 需要另行批准

- 非本專案管理元件的檔案（例如你自己的輸入圖片）。
- 需要遠端使用的敏感下載憑證（如私有 HF／CivitAI token）：未提供時回傳 `E_APPROVAL_REQUIRED`；**不會**用 stdin／base64／帶 token URL 等方式繞過。

### Models / 私有 repo

- 模型與外掛由 instance 直接下載，不經控制端轉傳。
- **v1 不轉傳任何憑證到 instance**：若 `config/models.yaml` 的條目帶 `auth_env`（私密來源），流程會在付費或長時間安裝**之前**以 `E_APPROVAL_REQUIRED` 失敗並列出該些條目；請改用公開來源。
- 未指定目的地前，本專案**不會**推送任何 repository、也不會擅自建立公開／私有 repo。

---

## 14. 費用、資料保存與 stop／destroy

- 時租上限只用於選機（整合 `dph_total`）；**流量費另計**，不是總花費上限。
- 正常流程成功後 instance 與 ComfyUI 持續運作；本工具**不會**自動關機。
- `stop`：保留資料，**仍持續收硬碟費**；重新啟動不保證取回原 GPU。
- `destroy`：**永久刪除 instance 內所有資料**（本工具 v1 不使用獨立 volume）。需要 `--confirm-destroy`，同意租用不等於同意刪資料。
- `stop`／`destroy` 也是**修改操作**：必須指定仍持有獨占權的 session（`--session-id`），執行前會先取得遠端操作鎖並驗證所有權；API 未確認成功時回報 `E_UPSTREAM`，不假裝已停止或已銷毀。
- 因為 instance 可能在命令被接受後立即失聯，`stop`／`destroy` 成功受理後**刻意保留**遠端鎖，等待平台轉換完成後由你或後續人工流程處理，不會先行解鎖。

```bash
vast-comfy --env-file .env instance stop    --instance <ID> --session-id <SID>
vast-comfy --env-file .env instance destroy --instance <ID> --session-id <SID> --confirm-destroy
```

- 程式結束、session 釋放、本機關機都不代表 instance 已停止或停止計費。
- 失敗時本工具不代你清理；`instance stop`／`destroy` 永遠由你明確呼叫。
- 重要圖片與工作流請定期 `download` 取回；`destroy` 前務必先下載。

---

## 15. 命令與 JSON 契約

### 錯誤碼（節錄）

| 錯誤碼 | 意義 |
| --- | --- |
| `E_CONFIG` | 設定／YAML／alias 格式錯誤 |
| `E_UNKNOWN_ALIAS` | 未知 alias（附候選提示，不自動修正） |
| `E_NO_OFFERS` / `E_PRICE_LIMIT` | 無合格 offer／全部超預算 |
| `E_OFFER_CHANGED` | 建立前重查發現 offer 已變 |
| `E_APPROVAL_REQUIRED` | 需要明確授權（付費／傳輸／憑證） |
| `E_CREATE_UNCERTAIN` | 建立結果不明（**未**重試） |
| `E_START_FAILED` / `E_START_TIMEOUT` | 啟動失敗／逾時（不自動處置） |
| `E_SSH_INFO_MISSING` / `E_SSH_AUTH` / `E_SSH_HOSTKEY` / `E_SSH_CONNECT` | SSH 相關失敗 |
| `E_LOCK_CONFLICT` / `E_LOCK_STALE` / `E_LOCK_JOBS_RUNNING` / `E_LOCK_NOT_HELD` | 鎖與所有權 |
| `E_JOB_FAILED` / `E_JOB_TIMEOUT` | job 失敗／等待逾時（job 可能仍在跑） |
| `E_INSTALL*` / `E_DEPLOY` | 安裝與部署問題（校驗／dirty／路徑） |
| `E_COMFY_BUSY` / `E_COMFY_UNAVAILABLE` | 佇列忙碌／服務異常 |
| `E_SMOKE_FAILED` | smoke 未達 READY |
| `E_INSTANCE_STATE` | instance 非 running／不存在 |
| `E_UPSTREAM` / `E_NETWORK` | Vast.ai API／網路問題 |

### Exit codes

| code | 意義 |
| --- | --- |
| 0 | 成功 |
| 1 | 一般失敗（看 JSON 的 `code`） |
| 2 | 使用方式錯誤 |
| 3 | 需要明確授權（例如缺 `--confirm-paid`） |
| 4 | 鎖／所有權衝突（含 stale） |
| 5 | 驗證未通過（smoke FAILED） |

### JSON 範例

```json
{
  "ok": false,
  "command": "flow create-and-prepare",
  "code": "E_APPROVAL_REQUIRED",
  "message": "Creating a paid instance requires explicit confirmation",
  "stage": "rent",
  "details": {"required_flag": "--confirm-paid", "dph_total_usd_per_hour": 0.41},
  "hints": ["Re-run with --confirm-paid only after reviewing the quote."]
}
```

成功結果一定是 `{"ok": true, "command": ..., ...}`。`--json` 模式下 stdout 只有這一個物件。

### 常用命令一覽

| 目的 | 命令 |
| --- | --- |
| 驗證設定 | `vast-comfy config validate` |
| alias | `vast-comfy gpu aliases` / `resolve` / `update-aliases` / `add-alias` |
| 報價 | `vast-comfy offers search --alias <A>` / `offers best --alias <A>` |
| 租用 | `vast-comfy rent create --alias <A> --offer-id <N> --label <L> --confirm-paid [--wait]` |
| 查證建立 | `vast-comfy rent verify --label <L>` |
| 實例 | `vast-comfy instance list` / `show` / `wait-ready` / `stop --session-id <SID>` / `destroy --confirm-destroy --session-id <SID>` / `logs` |
| 連線 | `vast-comfy ssh info` / `ssh test` / `ssh tunnel --start|--stop` |
| 金鑰 | `vast-comfy keys generate` / `register` / `list` |
| 部署 | `vast-comfy deploy package --instance <ID> --session-id <SID>`（僅在本專案管理元件；取得獨占權後才部署） |
| 鎖 | `vast-comfy session acquire` / `status` / `release` / `recover --accept-stale` |
| 工作 | `vast-comfy jobs start` / `status` / `logs` / `wait` / `list` |
| 安裝 | `vast-comfy install run` / `install verify` |
| 服務 | `vast-comfy comfy status` / `health` / `restart` / `verify-nodes` / `gpu-check` / `logs` |
| 驗收 | `vast-comfy smoke run` / `smoke report` |
| 下載 | `vast-comfy download report` / `images` / `outputs` |
| 組合 | `vast-comfy flow prepare --instance <ID>` / `flow create-and-prepare --alias <A> --confirm-paid` |

### Python API

```python
from vast_comfy.api import make_client, quote_for_alias_api, rent_and_prepare
from vast_comfy.config import load_config

cfg = load_config("my-config.yaml")
client = make_client(cfg)                      # 讀取 VAST_API_KEY
quote = quote_for_alias_api(cfg, "rtx_4060_ti_16gb", client=client)
result = rent_and_prepare(cfg, alias="rtx_4060_ti_16gb", label=None,
                          confirm_paid=True, client=client)
```

Python API 與 CLI 共用同一套驗證、硬條件檢查與執行邏輯；失敗丟 `VastComfyError`（含 `code`／`details`）。

---

## 16. 常見問題排解

| 症狀 | 原因與處理 |
| --- | --- |
| `Permission denied (publickey)` | 公鑰未在**建立 instance 之前**註冊；對既有 instance 補掛金鑰需用 Vast.ai 的 instance 專屬介面，或重建 instance |
| `E_SSH_HOSTKEY` | 專用 known_hosts 記錄的主機指紋變了。**不要**直接刪掉了事；先確認是否换了機器／proxy，再自行決定 |
| `E_LOCK_CONFLICT` | 有另一個 session 持有鎖。看 `session status` 的持有者資訊；到持有者的控制電腦結束其流程並確認遠端工作已停，再 `session recover --accept-stale` |
| 心跳過期訊息 | 只是回報。先確認舊持有者與其工作是否真的結束；沒有存活工作才能復原 |
| `E_COMFY_BUSY` | ComfyUI 有排隊中的生成工作。等它跑完；本工具不清佇列、不強制重啟 |
| `E_JOB_TIMEOUT` | 等待超時 ≠ 工作停止。用 `jobs status`／`jobs logs` 追蹤 |
| instance `offline`／`stopped` | v1 無法在無法驗證遠端鎖時安全操作；請在 Vast.ai 管理頁面處理，SSH 恢復後再回工具 |
| smoke 失敗看哪裡 | `smoke report --run-id <RUN>` 或 `jobs logs`；報告與圖片都在遠端 |

---

## 17. 模型授權與驗收範圍

- **Anima**（<https://huggingface.co/circlestone-labs/Anima>）：模型授權為 Circlestone Labs 的非商業授權（`circlestone-labs-non-commercial-license`）。本專案初始使用情境為**私人使用、自行生成圖片自行觀看**；收費託管模型服務需要另行確認授權。生成的圖片之商用條件請依模型頁面與其 LICENSE 為準。
- **Anima-LLLite 權重**（<https://huggingface.co/kohya-ss/Anima-LLLite>）與外掛（<https://github.com/kohya-ss/ComfyUI-Anima-LLLite>）同源授權，請見其 LICENSE。
- 驗收是**功能可用性**測試（能否實際出圖、節點與模型是否載入），不是美術品質保證，也不保證 LLLite 控制強度的主觀效果。

---

## 18. 驗證狀態

如實揭露（SPEC §17 真機驗收授權）：

**已在控制端完成**

- 單元測試＋模擬整合測試（本地 mock Vast.ai REST API／SSH／ComfyUI，**不接觸真實網路與 GPU**）：
  ```bash
  .venv\Scripts\python.exe -m pytest
  # 289 passed, 7 skipped（skipped 為 Linux-only 的 flock 實測）
  ```
- 覆蓋 SPEC §17 全 19 項：設定嚴格性、alias、硬條件逐項、整合報價與上限、付費授權、不可重試的建立、SSH 與主機指紋、鎖與心跳、job 逾時語意、下載校驗、白名單部署、smoke 判定（舊圖／單色／尺寸／缺節點／busy）、JSON／exit code 契約。
- 測試細節與範圍見 [`tests/README.md`](./tests/README.md)。

**尚未完成（不宣稱已可用）**

- 遠端 `flock` 競態與工作存活行為的**實機**驗證：測試已寫好並標記 `linux_only`（於 Linux 執行時自動啟用）；目前只在 Windows 上跳過。
- image 候選 `vastai/comfy:v0.40.0-cuda-12.9-py312`（amd64 digest `sha256:63ece7…`）**尚未**經本專案真機相容性驗證；實作以固定 tag／digest 為準。
- 真機付費驗收（完整 smoke suite 於真實 instance 出圖）需另外取得你的明確同意（GPU alias、報價、測試時間）。
- 在完成真機驗收之前，README 不會宣稱「已保證可用」。
