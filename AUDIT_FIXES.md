# 第二輪獨立審計修正紀錄

本輪以 `bc99bb0` 為基準。P1 表示可能破壞所有權、資料／憑證邊界或付費結果判讀，必須在真機前修正；P2 表示功能、狀態或設定契約錯誤，也納入本輪修正。`SPEC.md` 不變。

| 項目 | 嚴重性與判讀 | 修正／回歸證據 |
| --- | --- | --- |
| HTTP smoke 逾時後背景推論仍活，但管理程序退出 | P1：管理 PID 結束不代表 ComfyUI 工作結束。真正 flock 會在最後持 FD 程序退出後消失，不能只改 JSON 提示 | 持久 submission intent、背景 watcher、sticky timeout、獨立 progress；未取得可信終止證據前不解除保護；`session resolve-inference --confirm-verified` 是人工查證後的唯一解除路徑。`test_audit2_inference.py`／Linux 專用測試 |
| 共用 SSHRunner 的跨執行緒操作誤認為巢狀 | P1：兩個操作只取得一次遠端 lease，破壞同 session 排他 | 本機非阻塞 gate＋thread-local token；同執行緒真正巢狀共用 handle，inner retain 向外傳遞。`test_audit2_controller_locks.py` |
| stop／destroy 結果不明仍結束操作 | P1：請求可能已接受，重試／新工作可能與平台轉換重疊 | dispatch 前 retain；錯誤保留 lease、action outcome、instance／session／計費資訊；不重送。`test_audit2_controller_locks.py` |
| 父程序覆寫背景 job 終態 | P2：atomic replace 不能解決兩個 writer 的 read-modify-write 競態 | spawn 後 status 僅由 runner 寫入；progress 使用另一份檔案。遠端回歸測試 |
| 遠端 smoke 主入口沒有操作 guard | P2：正常 managed job 原本有外層鎖，但直接工具入口可繞過 | 入口亦驗證操作；正常長工作仍由 managed job 持鎖。遠端回歸測試 |
| run_id／下載路徑越界 | P1：識別符、報告 metadata 或符號連結可能越出授權目錄 | 可攜單段 ID、POSIX component 驗證、本機 resolve 包含性、遠端 realpath 檢查；整批預驗證，逐檔同目錄暫存及原子提升。`test_audit2_download.py`／`test_audit2_manifests.py` |
| 模型 ID 越出 downloads、模型目的地／發布競態 | P1：不受限 ID 與先檢查後 replace 可能改寫其他資料 | 控制端及遠端各自驗證 ID、來源與目的地；symlink 包含性；no-clobber hard link 發布。跨檔案系統先建立已驗證 staging。`test_audit2_installer.py` |
| URL 憑證繞過 auth_env default-deny | P1：query／userinfo 仍可把秘密送到遠端／日誌 | 來源只接受無憑證 HTTPS；拒 query、userinfo、fragment、編碼繞路；日誌另遮蔽。遠端重新驗證整批清單。`test_audit2_manifests.py`／`test_audit2_installer.py` |
| 畸形 create 回應誤判／裸 ValueError | P1：可能誘使操作者重送並重複付費 | 僅精確 success=true＋合法正整數 ID 算建立成功；缺失／畸形／矛盾回應一律 E_CREATE_UNCERTAIN，保留 label、送出前時間及可能計費資訊。明確 success=false 才當拒絕。`test_audit2_platform.py` |
| queue={} 被當成 idle | P1：無法確認沒有工作卻允許 restart | queue_running、queue_pending 必須都存在且為 list；格式未知 fail closed，不重啟。`test_audit2_platform.py` |
| 單獨租機入口只嘗試一次 SSH，錯誤缺少資源資訊 | P2：平台 running 不等於 SSH ready，並可能隱藏已付費資源 | CLI／API 共用有上限的 SSH readiness；post-create 例外均附 instance、label、計費提示。狀態儲存失敗亦不遺失參照。`test_audit2_platform.py` |
| 純 ssh-agent 身分遭拒 | P2：明列支援的驗證方式不能使用 | 未選定私鑰時納入 agent 公鑰；明確指定私鑰時不改用無關 agent 身分。`test_audit2_platform.py` |
| CLI 覆蓋 release_on_success、service 未傳遞 | P2：API／CLI 與明確設定不一致 | 未提供 keep-session 時保留設定決策；status／restart／logs 傳入設定 service。`test_audit2_platform.py` |
| 模型／插件 YAML 與固定版本驗證不足 | P2：錯誤可能在付費／修改遠端後才出現 | duplicate key／unknown field／重複 ID 及目的地拒絕；固定 SHA、嚴格 size／source schema；run/case 參數在副作用前驗證。`test_audit2_manifests.py` |

## 驗證邊界

- 所有本輪 API、SSH、ComfyUI 請求均為測試替身，沒有租機或真實 stop／destroy。
- Windows 測試不能證明 Linux kernel flock／FD 行為；Linux-only 測試需另在合適的本機／既有環境執行，不能因此自行租機。
- 下載提升是逐檔原子，不是整批交易。不支援 hard link 的目的檔案系統會安全拒絕；不宣稱可抵抗不受工具鎖約束的惡意 root 同時改換目錄。
- 推論提交結果不明、監看失聯或程序意外終止時，寧可保留待查證狀態，也不假裝工作已結束；解除需要操作者明確查證後使用 `session resolve-inference --confirm-verified`，這可能需要人工接手。
- 測試執行總數及尚未執行項目以本輪最終交付結果為準，不沿用前輪的測試數字。
