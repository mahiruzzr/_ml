# 🧠 nn0.py — 自製微型自動微分與深度學習引擎

這是一個純 Python 實現的微型深度學習底層框架。它不依賴 PyTorch 或 NumPy，僅用最基礎的 Python 語法與 `math` 函式庫，就優雅地重現了現代神經網路運作的核心基石。

---

## 🚀 核心模組解析

### 1. 標量自動微分引擎 (`Value`)
`Value` 類別是整個框架的靈魂。它實作了 **反向模式自動微分（Reverse-mode Automatic Differentiation）**。

* **動態計算圖**：每當你對 `Value` 對象進行數學運算（如 `+`, `*`, `exp`, `log`）時，它都會即時建立一個有向無環圖（DAG）。
* **局部梯度（Local Gradients）**：在相加或相乘的當下，節點就會默默記下這步運算對微積分的貢獻（例如乘法 $y = a \cdot b$，則 $\frac{\partial y}{\partial a} = b$）。
* **拓撲排序（Topological Sort）**：當你呼叫 `loss.backward()` 時，系統會先對圖中所有節點進行拓撲排序，確保「順序正確地」從輸出端一路將梯度乘回輸入端（連鎖律 Chain Rule）。

### 2. 現代化優化器 (`Adam`)
不同於傳統的隨機梯度下降（SGD），`Adam`（Adaptive Moment Estimation）結合了：
1.  **動量（Momentum）**：一階矩（`m`），記住之前的進前方向，幫你衝過平坦的鞍點。
2.  **自適應學習率（RMSProp）**：二階矩（`v`），根據梯度的稀疏程度縮放每個參數的更新幅度。
3.  **偏差修正（Bias Correction）**：在訓練初期（`step_count` 較小），修正動量被初始化為 0 造成的偏差。
4.  **線性衰減（Linear Decay）**：在訓練循環 `gd()` 中，透過手動計算 `lr_override` 讓學習率隨著時間變長而縮小，使模型後期收斂更穩定。

### 3. 神經網路基本算子
* `linear(x, w)`：純手工業矩陣乘法，計算 $W \cdot x$。
* `rmsnorm(x)`：**Root Mean Square Normalization**。相較於 LayerNorm，它不減去均值，只除以均方根。這是現代大語言模型（如 LLaMA / Gemma）的標配，計算更省時。

---

## ⚔️ 關鍵亮點：數值穩定的 Cross Entropy

在 `nn0.py` 中，最精彩的細節莫過於 `cross_entropy` 的優化。

### 傳統做法的問題 (`cross_entropy_simple`)
如果直接將 `softmax` 的機率值丟進 `.log()`：
1. 若某個類別的預測機率極度逼近 `0`，`math.log(0)` 就會直接噴出 `-inf` 或是 `NaN`（計算崩潰）。
2. `softmax` 裡面的 `math.exp(x)` 遇到太大的數值也會直接**上溢位（Overflow）**。

### 數學魔法：Log-Sum-Exp Trick
現代框架會利用對數律展開公式：

$$-\log\left(\frac{e^{x_t}}{\sum_i e^{x_i}}\right) = \log\left(\sum_i e^{x_i}\right) - x_t$$

為了防止 $e^{x_i}$ 爆炸，我們在括號內外同時減去最大值 $M$（`max_val`）：

$$\log\left(\sum_i e^{x_i - M}\right) + M - x_t = \log\left(\sum_i e^{x_i - M}\right) - (x_t - M)$$

這就是程式碼中第 4 步採用的精準公式：
```python
# 避開了除法，且 exp 的參數永遠 <= 0，絕對不會 Overflow！
return log_sum_exp - target_logit_shifted
