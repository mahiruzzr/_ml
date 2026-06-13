# 課程：機器學習 -- 習題總匯

| 欄位 | 內容 |
|------|------|
| 學期 | 114 學年下學期 |
| 學生 | 張政榮 |
| 學號末兩碼 | 46 |
| 教師 | [陳鍾誠](https://www.nqu.edu.tw/educsie/index.php?act=blog&code=list&ids=4) |
| 學校科系 | [金門大學資訊工程系](https://www.nqu.edu.tw/educsie/index.php) |
| 課程教材 | [ccc114b/_ml](https://github.com/ccc114b/_ml) |
| 作業存放 | [mahiruzzr/_ml](https://github.com/mahiruzzr/_ml) |

---

## 習題概覽

| 習題 | 主題 | 語言 | 使用 AI | 原始 Issue |
|------|------|------|---------|-----------|
| **習題一** | Hill Climbing TSP | C++ | [Grok](https://grok.com/share/c2hhcmQtNA_51cc0c06-01c7-4f60-922c-5d68c3a5351a) | #1 |
| **習題二** | 手寫數字分類 | Python | Antigravity 2.0 / Opencode | #2 |
| **習題三** | 自製自動微分引擎 | Python | [Gemini](https://gemini.google.com/share/6aaf1a1d0c04) | #3 |
| **習題四** | microGPT Transformer | Python | Antigravity 2.0 / Opencode | #4 |
| **習題五** | AI Agent 安全控管 | Python | Antigravity 2.0 / Opencode | #5 |
| **習題六** | N-gram 語言模型 | Python | Antigravity 2.0 / Opencode | #6 |

> AI 工具：主要使用 **Antigravity 2.0** 與 **Opencode**，部分使用 Grok 與 Gemini。

---

## 習題連結

| 習題 | 完整檔案 |
|------|---------|
| **習題一 | Hill Climbing TSP** | [\HW1/\](https://github.com/mahiruzzr/_ml/tree/55f56b3701414d1fa0431bc7b08e66d97a442357/HW1)（\hw1.cpp\ + README） |
| **習題二 | 手寫數字分類** | [ŭ2.jpg\](https://github.com/mahiruzzr/_ml/blob/bd124d009678cc3b4eae9a5510020fd4ca10d572/5552.jpg)（圖片辨識） |
| **習題三 | 自動微分引擎** | [\HW2/\](https://github.com/mahiruzzr/_ml/tree/fc4da5ece656f2bbf104c53e273df4368cb89d05/HW2)（n0.py\ + README） |
| **習題四 | microGPT** | [\gpt/\](https://github.com/mahiruzzr/_ml/tree/9f87e2425ebb29b506ffc7d8cfe0c3dad5fc9a43/gpt)（\gpt.py\ + README） |
| **習題五 | AI Agent** | [\HW3/\](https://github.com/mahiruzzr/_ml/blob/74070abd815f21d634f2cea309aa84a455b43e7c/HW3/README.md)（\gent0.py\ + README） |
| **習題六 | N-gram 語言模型** | [\HW4/\](https://github.com/mahiruzzr/_ml/tree/a70af0f9fdbc29be169b451f300fecde2b8c24b7/HW4)（\lm.py\ + README） |

---

## 各習題說明

### 習題一 | Hill Climbing 旅行推銷員問題（C++）

- **資料夾**：[\HW1/\](https://github.com/mahiruzzr/_ml/tree/55f56b3701414d1fa0431bc7b08e66d97a442357/HW1)
- **AI 工具**：Grok → [對話紀錄](https://grok.com/share/c2hhcmQtNA_51cc0c06-01c7-4f60-922c-5d68c3a5351a)
- **語言**：C++
- **檔案**：\hw1.cpp- **核心概念**：爬山演算法（Hill Climbing）求解 TSP
  - 隨機交換兩個城市位置產生鄰居解
  - 接受較短（或相等）路線，直至停止條件
  - 使用「負距離」技巧將最小化問題轉為最大化框架
- **編譯執行**：\g++ hw1.cpp -o tsp_hillclimbing -std=c++11 && ./tsp_hillclimbing
### 習題二 | 手寫數字分類

- **檔案**：[ŭ2.jpg\](https://github.com/mahiruzzr/_ml/blob/bd124d009678cc3b4eae9a5510020fd4ca10d572/5552.jpg)
- **語言**：Python
- **內容**：手寫數字圖片辨識分類

### 習題三 | nn0.py 自製微型自動微分引擎（Python）

- **資料夾**：[\HW2/\](https://github.com/mahiruzzr/_ml/tree/fc4da5ece656f2bbf104c53e273df4368cb89d05/HW2)
- **AI 工具**：Gemini → [對話紀錄](https://gemini.google.com/share/6aaf1a1d0c04)
- **語言**：Python
- **核心模組**：
  - \Value\ 類別：反向模式自動微分（Reverse-mode AD）
  - \Adam\ 優化器：動量 + 自適應學習率 + 偏差修正
  - 神經網路算子：\linear\、msnorm  - 數值穩定的 Log-Sum-Exp Cross Entropy
- **不依賴 PyTorch / NumPy**，純 Python + \math\ 函式庫

### 習題四 | microGPT：從零實作 GPT（Python）

- **資料夾**：[\gpt/\](https://github.com/mahiruzzr/_ml/tree/9f87e2425ebb29b506ffc7d8cfe0c3dad5fc9a43/gpt)
- **語言**：Python
- **檔案**：\gpt.py\（約 200 行，零外部依賴）
- **六層架構**：
  1. **Dataset**：名字字元序列
  2. **Tokenizer**：Character-level（27 tokens）
  3. **Autograd Engine**：反向模式自動微分
  4. **模型架構**：Embedding + Multi-Head Self-Attention + MLP + Residual
  5. **訓練迴圈**：Forward → Backward → Adam 更新
  6. **推論**：Autoregressive Sampling with Temperature
- **訓練**：\python gpt.py input.txt
### 習題五 | agent0.py：AI Agent 安全控管（Python）

- **資料夾**：[\HW3/\](https://github.com/mahiruzzr/_ml/blob/74070abd815f21d634f2cea309aa84a455b43e7c/HW3/README.md)
- **語言**：Python
- **核心設計**：五層深度防禦（Defense in Depth）
  - Layer 1：Denied Tool List（黑名單命令）
  - Layer 2：Bash Classification（命令分類）
  - Layer 3：Interactive Prompt（使用者確認）
  - Layer 4：Workspace Path Scope（路徑逃逸防護）
  - Layer 5：LLM Safety Review（語義審查）
- **安全模式**：ead-only\ / \workspace-write\ / \danger-full-access\ / \prompt- **執行**：\ollama serve\ → \python3 agent0.py
### 習題六 | N-gram 語言模型（Python）

- **資料夾**：[\HW4/\](https://github.com/mahiruzzr/_ml/tree/a70af0f9fdbc29be169b451f300fecde2b8c24b7/HW4)
- **語言**：Python
- **檔案**：\lm.py\ + \	w.txt\（中文訓練語料）
- **核心概念**：
  - N-gram 計數 + Add-k Smoothing
  - Temperature Sampling + Top-k 抽樣
  - 與 Transformer 對比：固定 context vs 任意長度、計數 vs 梯度下降
- **執行**：\python3 lm.py tw.txt 2\（bigram）/ \python3 lm.py tw.txt 3\（trigram）

---

## 補充說明

- 各習題資料夾內皆有獨立的 \README.md\ 詳細說明
- 習題一、三使用外部 AI（Grok / Gemini），其餘預設使用 **Antigravity 2.0** 與 **Opencode**

---

*最後更新：2026 年 6 月*
