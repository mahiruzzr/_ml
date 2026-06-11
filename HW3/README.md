# agent0.py — AI Agent with Security Controls

一個具備記憶、工具執行與多層安全控管的 AI Agent，基於 Ollama LLM。安全架構參考 [claw-code](https://github.com/ultraworkers/claw-code)（Rust-based AI agent harness）的 Permission System。

## 快速開始

```bash
# 確保 Ollama 正在執行
ollama serve

# 執行 agent
python3 ~/.agent0/agent0.py

# 或從 .gemini 目錄執行
cd ~/.gemini && python3 agent0.py
```

## 權限模式

agent0 有四種安全模式，寫在 `~/.agent0/config.json`：

| 模式 | 值 | 允許的操作 |
|---|---|---|
| `read-only` | 0 | 只能讀檔、搜尋、ls、cat、grep，禁止所有寫入 |
| `workspace-write` | 1 | 可在 workspace 內讀寫，阻擋破壞性命令（rm -rf /） |
| `danger-full-access` | 2 | 無限制，跳過 LLM 審查 |
| `prompt` | 3 | 每次執行命令前詢問使用者 |

```json
{
  "permission_mode": "workspace-write",
  "workspace_roots": ["/home/user/project"],
  "denied_tools": ["rm", "dd", "mkfs", "shutdown"],
  "shell_timeout": 60
}
```

---

## 五層安全防禦體系

這是整個程式的核心設計。每一層都有明確的職責，由淺入深：

```
使用者請求
    │
    ▼
┌─────────────────────────────────────┐
│ Layer 1: Denied Tool List           │  ← 擋 rm, dd, sudo, chmod 等
│   檢查指令是否在黑名單中              │
└─────────────────────────────────────┘
    │ 通過
    ▼
┌─────────────────────────────────────┐
│ Layer 2: Bash Classification        │  ← 根據權限模式判斷
│   將指令分類為 read-only / write     │
│   / destructive / network / package  │
└─────────────────────────────────────┘
    │ 通過
    ▼
┌─────────────────────────────────────┐
│ Layer 3: Interactive Prompt         │  ← prompt 模式才觸發
│   詢問使用者是否允許執行              │
└─────────────────────────────────────┘
    │ 通過
    ▼
┌─────────────────────────────────────┐
│ Layer 4: Workspace Path Scope       │  ← 防止 .. 逃逸、symlink 攻擊
│   檢查所有路徑操作是否在 workspace 內  │
└─────────────────────────────────────┘
    │ 通過
    ▼
┌─────────────────────────────────────┐
│ Layer 5: LLM Safety Review          │  ← 讓另一個 LLM 做語義審查
│   用第二個模型判斷指令語意是否安全      │
└─────────────────────────────────────┘
    │ 通過
    ▼
   執行指令
```

### Layer 1 — Denied Tool List（`blocks_tool()`）

檢查指令的第一個詞（命令名）是否在黑名單中：

```python
denied_tools = ["rm", "dd", "mkfs", "fdisk", "shutdown", "reboot", ...]
denied_prefixes = ["sudo ", "su ", "chmod 777", "chown ", "passwd"]
```

如果 `cat /etc/shadow`，第一詞是 `cat`，不在黑名單 → 通過。
如果 `rm -rf /tmp/*`，第一詞是 `rm`，在黑名單 → 阻擋。

### Layer 2 — Bash Classification（`classify_command()` + `check_bash()`）

將 shell 指令分類為：

```python
# 檢查 shell metacharacters (;, |, &, $, `, >, <, (, ), \n)
has_meta = any(c in cmd for c in ";|&$`><()\n")

# 檢查破壞性模式
if re.search(r"\brm\s+-rf\s*/", cmd):
    return ("destructive", "rm -rf on root")

# 檢查唯讀命令
if first_word in READ_ONLY_COMMANDS and not has_meta:
    return ("read-only", None)
```

然後根據權限模式決定是否允許：

```python
if mode == READ_ONLY and classification != "read-only":
    return (False, "read-only mode: write not allowed")
if mode == WORKSPACE_WRITE and classification == "destructive":
    return (False, f"destructive blocked: {warning}")
```

### Layer 3 — Interactive Prompt

當模式設為 `prompt` 時，每條命令都問使用者：

```
  needs approval: rm -rf /tmp/build
  allow? (y/N): n
  denied
```

### Layer 4 — Workspace Path Scope（`WorkspaceScope`）

改寫自 claw-code-main 的 `path_scope.py`。他的核心是：

1. 從 payload 中**萃取路徑候選**（用 `shlex.split` 解析引號、展開 `$VAR` 和 `~/`）
2. 解析每個候選路徑到**絕對路徑**
3. 處理 glob 展開（`*.txt` → 多個檔案）
4. 檢查**解析後的路徑**是否在 workspace root 底下

```python
scope = WorkspaceScope(["/home/user/project"])
decision = scope.validate("cat ../etc/passwd")
# allowed=False, reason="path outside workspace"
```

它會擋住：
- `../` 逃逸
- symlink 指向外部
- glob 展開到外部檔案
- `$HOME/../../etc` 環境變數展開逃逸

### Layer 5 — LLM Safety Review

前四層都是**語法/規則層面**的過濾。第五層用**語義理解**：

```python
def review_command(cmd):
    # 用另一個 LLM 模型判斷語意是否安全
    # 輸出 "SAFE" 或 "UNSAFE - reason"
```

這是 defense in depth 的最後防線，用來擋住規則無法涵蓋的攻擊（例如 `python3 -c "import os; os.remove('/etc/passwd')"`）。

---

## Config 系統

設定檔案階層（參考 claw-code-main 的 5 層 config）：

```
~/.agent0/config.json          ← agent0 使用的單一設定檔
```

可設定的項目：

| 欄位 | 預設值 | 說明 |
|---|---|---|
| `permission_mode` | `"workspace-write"` | 安全模式 |
| `workspace_roots` | `[cwd]` | 允許操作的工作目錄列表 |
| `denied_tools` | `["rm", "dd", ...]` | 黑名單指令 |
| `denied_prefixes` | `["sudo ", ...]` | 黑名單前綴 |
| `shell_timeout` | `60` | 指令超時秒數 |

---

## 記憶系統

對話記錄採用 XML-like 格式保存，提供給 LLM 作為 context：

```xml
<memory>
  <item>使用者叫小明</item>
  <item>專案路徑是 ~/myproject</item>
</memory>

<history>
  <user>幫我建立一個資料夾</user>
  <assistant>沒問題，我先檢查路徑</assistant>
  <tool>$ mkdir test</tool>
</history>
```

- 記憶上限：`MAX_TURNS = 5` 輪，超過自動淘汰舊對話
- 關鍵資訊由 LLM 自動萃取後存入 `key_info` 清單
- 可用 `/memory` 指令查看

---

## 指令列表

| 指令 | 功能 |
|---|---|
| `/quit` 或 `/exit` | 結束 |
| `/memory` | 顯示長期記憶 |
| `/mode` | 顯示目前權限模式 |
| `/config` | 顯示目前設定 |

---

## 程式結構

```
agent0.py (~600 行)
├── Configuration           # 路徑、模型名稱、設定檔
├── PermissionMode          # 四種安全模式（IntEnum）
├── AgentConfig             # JSON 設定檔的載入/儲存
├── WorkspaceScope          # 路徑範圍驗證（from claw-code）
├── Bash Classification     # 指令分類系統
├── ToolPermissionContext   # 權限檢查閘門
├── Memory                  # 對話記憶管理
├── Ollama API              # 用 urllib 呼叫 Ollama（零依賴）
├── Safety Review           # LLM 語義審查
├── UI Helpers              # 使用者互動
└── main loop               # Agent 主迴圈
```

---

## 與 claw-code-main 的對應關係

| agent0.py | claw-code (Rust) / src/ (Python) | 差異 |
|---|---|---|
| `PermissionMode` | `rust/crates/runtime/src/permissions.rs` | 相同 enum，少了 `Allow` 模式 |
| `AgentConfig` | `rust/crates/runtime/src/config.rs` | 簡化版（只有 2 層，claw 有 5 層） |
| `WorkspaceScope` | `src/path_scope.py` | 移植版，功能相同但簡化 |
| `ToolPermissionContext` | `src/permissions.py` | 合併了 blocks + check 邏輯 |
| `classify_command` | `rust/.../bash_validation.rs` | 簡化版 |
| `review_command` | 無對應 | agent0 獨有的 LLM 語義審查層 |

主差別：claw-code 是 production-grade（Rust、非同步、sandbox、plugin 系統），agent0 是純 Python 教育版。

---

## 自訂安全規則

修改 `~/.agent0/config.json`：

```json
{
  "permission_mode": "read-only",
  "workspace_roots": ["/home/user/project", "/tmp/work"],
  "denied_tools": ["rm", "mv", "cp", "dd", "mkfs", "curl", "wget"],
  "denied_prefixes": ["sudo ", "chmod ", "chown "],
  "shell_timeout": 30
}
```

或直接編輯 `.py` 中的 `AgentConfig` dataclass 的預設值。

---

## 概念總結

**Defense in depth（深度防禦）**是 agent0 安全架構的核心思想：

- **規則層（1-4）**負責過濾已知威脅——語法簡單、速度快、確定性高
- **語義層（5）**負責過濾未知威脅——需要 LLM 語義理解、速度慢

claw-code-main 的啟發：production 的 AI agent 安全不是靠「LLM 自己乖不乖」，而是靠一層層不可繞過的機制閘門，每一層都假定「前一層被繞過」來設計。
