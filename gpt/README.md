# microGPT — 從零實作 GPT 學習筆記

一個 200 行的純 Python GPT 實作，零外部依賴。完整包含：autograd engine、Transformer 架構、Adam 優化器、訓練迴圈、推論生成。

## 目錄結構

```
gpt/
├── gpt.py       # 主程式（訓練 + 推論）
├── input.txt    # 訓練數據（每行一個名字）
└── README.md    # 本文件
```

## 快速開始

```bash
# 下載名字資料集
curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

# 訓練並生成
python gpt.py names.txt
```

約 30 秒後你會看到：

```
dataset: names.txt, num docs: 32033
vocab size: 27 (26 letters + BOS)
num params: 4192
step    1 /  500 | loss 3.3660
step   51 /  500 | loss 2.6123
...

--- generated names ---
  kamon
  ann
  karai
  jaire
  ...
```

---

## 六層洋蔥：從上往下理解

### Layer 1 — 資料（Dataset）

**檔案位置：`gpt.py:6-17`**

每個名字是一個「document」。模型的工作是學習名字的字元統計規律，然後生成新的、看起來像真的名字。

```
emma      → [BOS, e, m, m, a, BOS]    ← 預測下一個字元
oliva     → [BOS, o, l, i, v, a, BOS]
```

概念連結：ChatGPT 裡你跟它的對話也是一個「document」——你的 prompt 是前半段，它的回覆是後半段。模型只是在做 document completion。

---

### Layer 2 — Tokenizer

**檔案位置：`gpt.py:19-24`**

將字串轉成整數 ID。這裡用最簡單的 character-level：

```python
uchars = ['a', 'b', ..., 'z']
BOS = 26                                  # 特殊分隔記號
vocab_size = 27                           # 總共 27 個 token
```

BOS（Beginning Of Sequence）告訴模型「名字開始/結束了」。

對比 production（GPT-4）：使用 subword tokenizer（BPE），vocab size 約 100K，常用字（"the"）變成單一 token。

---

### Layer 3 — Autograd Engine（Value class）

**檔案位置：`gpt.py:29-77`**

這是整套系統最核心的數學引擎。每個 `Value` 物件做的事：

```
Value
├── .data      ← 前向傳播的數值
├── .grad      ← 反向傳播的梯度
├── _children  ← 計算圖的子節點
└── _local_grads  ← ∂(output)/∂(child_i)
```

#### 前向傳播（Forward）

你寫 `a * b` 時，它回傳一個新的 `Value`：

```python
def __mul__(self, o):
    return Value(
        data = self.data * o.data,   # 計算結果
        children = (self, o),         # 記住來源
        local_grads = (o.data, self.data)  # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    )
```

#### 反向傳播（Backward）

當你呼叫 `loss.backward()`：

1. **拓樸排序**：從 loss 開始，找出計算圖的正向依賴順序
2. **設定 seed**：`self.grad = 1`，因為 ∂loss/∂loss = 1
3. **鏈式法則**：反向遍歷，對每個節點：

```
child.grad += local_grad × parent.grad
         ↑           ↑            ↑
   累積梯度    ∂(parent)/∂(child)   ∂(loss)/∂(parent)
```

為什麼用 `+=`（累積）？因為當一個節點被多條路徑使用時，梯度要加起來（multivariable chain rule）：

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c = 6.0
L = c + a       # L = 8.0
L.backward()
print(a.grad)   # 4.0  ← 來自 c 的路徑 (b=3) + 來自 L 的路徑 (1)
```

支援的操作與其 local gradients：

| Op | Forward | ∂/∂input |
|---|---|---|
| `a + b` | a + b | 1, 1 |
| `a * b` | a × b | b, a |
| `a ** n` | aⁿ | n·aⁿ⁻¹ |
| `log(a)` | ln(a) | 1/a |
| `exp(a)` | eᵃ | eᵃ |
| `relu(a)` | max(0,a) | 1 if a>0 else 0 |

---

### Layer 4 — 模型架構（mini GPT）

**檔案位置：`gpt.py:82-156`**

#### Embedding Layer

Token ID 5 不能直接餵給神經網路——它只是個索引。所以我們用一個 lookup table 把它轉成向量：

```python
tok_emb = state_dict['wte'][token_id]  # token → 8 維向量
pos_emb = state_dict['wpe'][pos_id]    # 位置 → 8 維向量
x = tok_emb + pos_emb                  # 混合「什麼字」+「在哪個位置」
```

#### Multi-Head Self-Attention

Attention 是 token 之間**溝通**的機制：

```
Query:    "我在找什麼？"
Key:      "我有什麼？"
Value:    "如果被選中，我提供什麼？"
```

每個 attention head 獨立計算：

```python
scores = Q · Kᵢ / √d       # 計算 query 跟每個過去 token 的匹配度
weights = softmax(scores)   # 轉成機率
output = Σ weightsᵢ × Vᵢ   # 加權總和
```

多頭的好處：不同 head 學到不同模式（一個關注母音、一個關注字首、一個關注結尾）。

#### MLP（Feed-Forward）

Attention 之後，每個 token 獨立「思考」：

```python
x = linear(x, W1)    # 投影到 2× 維度（16 維）
x = relu(x)          # 非線性激活
x = linear(x, W2)    # 投影回原維度（8 維）
```

Transformer 的節奏：**溝通（Attention）→ 思考（MLP）→ 溝通 → 思考 → ...**

#### Residual Connection

每一層的輸出 = 層輸出 + 原始輸入。這讓梯度可以直接流過，是深層網路能訓練的關鍵。

---

### Layer 5 — 訓練迴圈

**檔案位置：`gpt.py:160-192`**

#### Forward

對每個名字，逐字元預測下一個字元：

```
[BOS, e, m, m, a, BOS]
  ↓    ↓  ↓  ↓  ↓   ↓
預測: e  m  m  a  BOS
```

每個位置的 loss 是 cross-entropy：

```python
loss = -log(p(target_token))  # 模型對正確答案有多「驚訝」
```

- 隨機猜測（27 中 1）：loss = -log(1/27) ≈ 3.3
- 完美預測：loss = -log(1) = 0

#### Backward

`loss.backward()` 跑完整個計算圖的反向傳播，每個參數的 `.grad` 告訴你「往哪邊調 loss 會降低」。

#### Adam Optimizer

比單純的 gradient descent 聰明：

```python
m = β₁·m + (1-β₁)·grad      # 動量（累積方向）
v = β₂·v + (1-β₂)·grad²    # 適應性學習率（大梯度 → 小步）
p -= lr · m̂ / (√v̂ + ε)      # 更新參數
```

- `β₁ = 0.85`：momentum 衰減率
- `β₂ = 0.99`：學習率適應衰減率
- `m̂`, `v̂`：bias correction（因為 m, v 初始為 0）

---

### Layer 6 — 推論（Autoregressive Sampling）

**檔案位置：`gpt.py:196-213`**

訓練完後固定參數，從 BOS 開始重複：

1. 餵入目前 token → 得到 27 個 logits
2. logits / temperature → softmax → 機率分布
3. 根據機率採樣一個 token
4. 如果抽到 BOS → 名字結束
5. 否則把抽到的 token 當作下一步輸入

Temperature 控制隨機性：

| T | 效果 |
|---|---|
| 0 | 永遠選最高機率 token（很無聊但穩定） |
| 0.5 | 適度隨機（預設值） |
| 1.0 | 照學到的分布採樣 |
| >1 | 分布趨平，輸出更多樣但也更亂 |

---

## 關鍵數字

| 項目 | microGPT | GPT-2 | GPT-4 |
|---|---|---|---|
| 參數量 | 4,192 | 1.6B | ~1.8T |
| Vocab size | 27 | 50,257 | ~100K |
| Embed dim | 8 | 768 | ~16K |
| Layers | 1 | 12 | ~120 |
| Context | 16 | 1,024 | 128K |
| 資料量 | 32K 名字 | 40GB 網路文字 | 數十 TB |
| 訓練時間 | 30 秒 (CPU) | 數天 (GPU) | 數月 (萬顆 GPU) |

---

## 跟真實 LLM 的差距

理解這 200 行之後，production 工程只是在 scale up：

| 面向 | microGPT | Production LLM |
|---|---|---|
| **資料** | 32K 名字 | 數兆 token，去重、過濾、混合 |
| **Tokenizer** | Character-level (27) | BPE subword (~100K) |
| **硬體** | CPU, 純 Python | GPU/TPU, CUDA FlashAttention |
| **精度** | float64 Python | float16 / int8 / fp8 |
| **最佳化** | Adam + linear decay | 複雜 scheduler + weight decay + warmup |
| **訓練後** | 無 | SFT + RLHF (DPO/PPO) |
| **推論** | 單一序列 | batching + KV cache paging + speculative decoding |
| **架構** | 精簡 GPT | RoPE, GQA, MoE, gated activations |

演算法本質完全一樣——剩下的都是效率。

---

## 自行探索

```bash
# 改參數試試
python gpt.py input.txt              # 用預設 names.txt
python gpt.py your_data.txt          # 用自己的資料
```

可以玩的參數：

- `n_embd`（16/32/64）：更大的 embedding → 更好結果
- `n_head`（4/8）：更多 attention head
- `n_layer`（2/3/4）：更深網路
- `num_steps`（1000/2000）：訓練更久
- `temperature`（0.1/0.8/1.5）：控制隨機性

---

## 參考文獻

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017
- [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Andrej Karpathy, 2026
- [micrograd](https://github.com/karpathy/micrograd) — Karpathy's autograd engine
- [makemore](https://github.com/karpathy/makemore) — character-level language models
- [nanoGPT](https://github.com/karpathy/nanoGPT) — minimal GPT-2 reproduction
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy 教學影片
