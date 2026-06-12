# N-gram 語言模型（非 Transformer）

## 概念說明

這是一個**不使用 Attention/Transformer** 的傳統語言模型，核心概念與 GPT 完全相同：

```
GPT 生成流程：
  ① 給定 prompt（前綴）
  ② 預測下一個字的「機率分布」
  ③ 從分布中「抽樣」一個字
  ④ 把新字加到序列尾端，回到步驟②
  ⑤ 重複直到產生 <EOS> 或達到最大長度
```

**唯一差別**：GPT 用 Transformer 算機率，這裡用 **N-gram 計數 + Smoothing**。

## 檔案結構

```
/home/kiyotaka/.gemini/
├── lm.py       # N-gram 語言模型主程式
├── tw.txt      # 中文訓練語料（147行短句）
└── README.md   # 本文件
```

## 執行方式

```bash
cd /home/kiyotaka/.gemini
python3 lm.py tw.txt 2   # bigram
python3 lm.py tw.txt 3   # trigram
```

互動模式：直接輸入文字接龍，`q` 離開。

## 抽樣策略

| 參數 | 作用 |
|------|------|
| temperature | 越高->越隨機有創意，越低->越保守 |
| top_k | 只從前 k 個高機率字抽樣，避免罕字亂入 |

## 原理簡述

### 1. N-gram 計數
n=2 時，滑動視窗統計：「小貓」->「坐」的次數。
機率 = count(context + char) / count(context)

### 2. Add-k Smoothing
未出現的組合機率不為零：P = (count + k) / (total + k * V)

### 3. Temperature Sampling
logit = log(P) / temp -> softmax -> 抽樣

## 與 Transformer 的對比

| 特性 | Transformer | N-gram |
|------|-------------|--------|
| 上下文長度 | 任意 | 固定 n |
| 訓練 | 梯度下降 | 直接計數 |
| 泛化 | 強（可理解新組合） | 弱（未見組合靠 smoothing） |
| 參數量 | 百萬以上 | V^n（稀疏） |

## 限制

- context 長度固定（n=2 只看前 1 字）
- n 越大資料越稀疏
- 無語意理解，純統計共現

## 延伸方向

1. 增加 n（但小語料會稀疏）
2. Back-off N-gram（高階無資料時降階）
3. Log-Linear 模型（CountVectorizer + LogisticRegression）
4. RNN/LSTM（可變長度 context，Transformer 前主流）

## 參考

- 原始碼：https://github.com/ccc114b/cccocw
- Jurafsky & Martin -- Speech and Language Processing Ch.3
