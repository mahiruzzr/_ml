# Face Swap Project — AI 換臉實作

> **AI 使用聲明**：本作業使用 AI（GitHub Copilot / Claude）輔助生成部分程式碼架構及此份文件，核心演算法邏輯（insightface API 呼叫、pipeline 設計）為本人理解後實作。
>
> **參考來源**：Deep-Live-Cam（架構靈感）、insightface 官方範例。
>
> **原創聲明**：本作業為原創，所有程式碼皆由本人理解後撰寫，無直接複製他人程式碼。

---

## 專案結構

```
~/.gemini/face-swap-project/
├── images/                    # 測試圖片 / 影片
│   ├── source.jpg             #   來源人臉（要複製的臉）
│   ├── target.jpg             #   目標人臉（被換掉的脸）
│   └── test_video.mp4         #   測試用影片（5秒, 512x512）
├── output/                    # 輸出結果
│   ├── result.jpg             #   v1 靜態換臉結果
│   ├── result_v2.mp4          #   v2 影片換臉結果
│   ├── result_v2_enhanced.mp4 #   v2 開啟增強
│   └── v3_test_poisson.jpg    #   v3 Poisson blending 測試
├── v1/
│   └── swap_face.py           # 靜態圖片換臉
├── v2/
│   ├── __init__.py
│   └── swap_video.py          # 影片換臉
├── v3/
│   ├── __init__.py
│   └── camera_swap.py         # 即時攝影機換臉
├── test_imports.py            # 環境測試腳本
└── README.md
```

---

## 環境建置

### WSL（Ubuntu）

```bash
# 1. 建立虛擬環境
cd ~/.gemini/face-swap-project
python3 -m venv .venv

# 2. 啟動虛擬環境並安裝套件
source .venv/bin/activate
pip install insightface onnxruntime opencv-python numpy

# 3. 驗證安裝（所有套件 import 成功會印出 "全部OK!"）
.venv/bin/python test_imports.py

# 4. 首次執行時，insightface 會自動下載 buffalo_l 模型（~300MB）
#    以及手動下載 inswapper_128.onnx（~554MB）
```

### 模型檔案位置

```
~/.insightface/models/
├── buffalo_l/              # 自動下載：RetinaFace + ArcFace 等
│   ├── det_10g.onnx        #   人臉偵測模型
│   ├── w600k_r50.onnx      #   人臉辨識模型（ArcFace）
│   ├── 1k3d68.onnx         #   3D 關鍵點模型
│   ├── 2d106det.onnx       #   2D 關鍵點模型
│   └── genderage.onnx      #   性別年齡預測
└── inswapper_128.onnx      # 手動下載：換臉生成模型（554MB）
```

---

## v1 — 靜態圖片換臉

### Pipeline

```
source.jpg → 載入圖片 → 偵測人臉 → 提取特徵(512维 embedding)
                                                    ↓
target.jpg → 載入圖片 → 偵測人臉 → 臉部替換(inswapper) → 輸出 result.jpg
```

### WSL 執行指令

```bash
# 基本用法
.venv/bin/python v1/swap_face.py \
    --source images/source.jpg \
    --target images/target.jpg \
    --output output/result.jpg
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--source` | (必填) | 來源人臉圖片路徑，此人的臉會被複製 |
| `--target` | (必填) | 目標圖片路徑，此圖的人臉會被替換 |
| `--output` | `output/result.jpg` | 輸出圖片路徑 |

### 實測結果

| 項目 | 數據 |
|------|------|
| 輸入圖片 | source.jpg (1280x886, 6人) / target.jpg (512x512, 1人) |
| 處理時間 | ~5 秒 |
| 輸出 | result.jpg (512x512, 97KB) |

---

## v2 — 影片換臉

### Pipeline

```
input.mp4 → 解碼影片 → 逐 frame 換臉 → [臉部增強] → 編碼影片 → output.mp4
```

### WSL 執行指令

```bash
# 基本：每幀都做換臉
.venv/bin/python v2/swap_video.py \
    --source images/source.jpg \
    --target input.mp4 \
    --output output/result.mp4

# 跳幀加速：每 3 幀做一次，中間沿用上次結果
.venv/bin/python v2/swap_video.py \
    --source images/source.jpg \
    --target input.mp4 \
    --every-nth 3

# 全開：跳幀 + LAB 色彩匹配增強
.venv/bin/python v2/swap_video.py \
    --source images/source.jpg \
    --target input.mp4 \
    --enhance --every-nth 3
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--source` | (必填) | 來源人臉圖片路徑 |
| `--target` | (必填) | 目標影片路徑（.mp4 / .avi 等 OpenCV 支援格式） |
| `--output` | `output/result.mp4` | 輸出影片路徑，格式由副檔名決定 |
| `--det-size` | `640` | 人臉偵測解析度（越高越準越慢，越低越快越容易漏） |
| `--every-nth` | `1` | 每 N 幀做一次換臉，其餘沿用上一幀結果（1=每幀都做） |
| `--enhance` | 不啟用 | LAB 色彩匹配 + 高斯模糊邊緣融合，讓換臉更貼合目標膚色 |

### 運作原理

- `--every-nth 3`：frame 0 做換臉，frame 1~2 直接複製 frame 0 的結果
  - 如果目標影片中人物移動不大，人眼幾乎看不出跳幀
  - 速度可提升 2~3 倍
- `--enhance`：將換臉後的臉部區域轉換到 LAB 色彩空間
  - 分別對 L（亮度）、A（綠紅）、B（藍黃）三個通道做 histogram matching
  - 讓換上的臉的色調、亮度貼近目標幀的原始膚色

### 實測結果（CPU, 512x512, 50 frames, 10 FPS）

| 模式 | 耗時 | 速度 | 備註 |
|------|------|------|------|
| 預設 (`--every-nth 1`) | 110.4s | 0.5 fps | 每幀都跑 model |
| 跳幀 + 增強 | 44.1s | 1.1 fps | 快 2.5 倍 |

---

## v3 — 即時攝影機串流

### Pipeline

```
Camera → 擷取 frame → 逐 frame 換臉 → [Poisson blending] → 顯示 + FPS
```

### WSL 執行指令（需有 camera passthrough，建議直接在 Windows 執行）

```bash
# Windows PowerShell (非 WSL)：
cd C:\Users\zhangzhengrong\OneDrive\桌面\face-swap-project
python v3\camera_swap.py --source images/source.jpg

# 啟用 Poisson blending
python v3\camera_swap.py --source images/source.jpg --poisson

# 降低偵測解析度 + 跳幀（更快但較不穩）
python v3\camera_swap.py --source images/source.jpg --det-size 240 --every-nth 5
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--source` | (必填) | 來源人臉圖片路徑 |
| `--camera` | `0` | 攝影機裝置 ID（0=預設鏡頭，1=外接鏡頭） |
| `--det-size` | `320` | 人臉偵測解析度（v3 預設比 v2 低，因為要即時） |
| `--every-nth` | `3` | 每 N 幀做一次換臉，其餘沿用上次結果（v3 預設跳幀） |
| `--display-scale` | `1.0` | 顯示視窗縮放比例（0.5=半尺寸） |
| `--poisson` | 不啟用 | Poisson blending：用 `cv2.seamlessClone()` 做邊界融合 |

### 快速鍵

| 按鍵 | 功能 |
|------|------|
| `Q` | 離開程式 |

### Poisson blending 原理

```python
# cv2.seamlessClone(前景, 背景, 遮罩, 中心點, 融合模式)
cv2.seamlessClone(swapped_bgr, original_bgr, face_mask, center, cv2.NORMAL_CLONE)
```

Poisson blending 不是簡單的 alpha 混合，而是解一個 Poisson 方程式：
- 保留前景（換臉結果）的**梯度**（細節、紋理）
- 但在邊界處強制匹配背景（原始畫面）的**顏色**
- 結果：邊界無縫過渡，沒有明顯的貼圖接縫

---

## 各版本比較

| 特性 | v1 | v2 | v3 |
|------|----|----|----|
| 輸入 | 單張圖片 | 影片檔 | 攝影機即時串流 |
| 輸出 | 單張圖片 | 影片檔 | 即時顯示 |
| 核心模型 | insightface inswapper_128 | 同 v1 + LAB enhance | 同 v1 + Poisson blending |
| 跳幀加速 | - | `--every-nth` | `--every-nth`（預設 3） |
| 預設 det_size | 640 | 640 | 320 |
| 處理速度 (CPU) | 即時 (~5s/張) | 0.5~1.1 fps | 相依於 camera + CPU |

---

## v4 — 優化加速（已實作）

### 優化項目

| 優化 | v2/v3 問題 | v4 解決方案 |
|------|-----------|------------|
| ONNX provider 切換 | 硬編碼 CPU | `--provider` 參數支援 cpu / cuda / dml / tensorrt / openvino |
| Face tracking | 每幀都要做完整偵測 | `--track-interval N`：每 N 幀才跑偵測，中間沿用上一幀的人臉位置 |
| 多解析度策略 | 單一 det_size，漏臉就沒了 | `--multi-res`：若目前解析度沒找到臉，自動切換其他解析度重試 |

### Pipeline

```
輸入 → [每 N 幀] 人臉偵測 → 換臉 → 輸出
                ↑ 沿用上幀位置（其餘幀）
```

### 核心元件：`v4/processor.py`

`OptimizedFaceProcessor` 類別封裝了 v4 的所有優化：

- `FaceTracker`：控制人臉偵測頻率，非偵測幀回傳快取的人臉資料
- `switch_provider()`：動態切換 ONNX ExecutionProvider
- `multi_res` 模式：嘗試 (320,320) → (640,640) → (192,192) 三種偵測解析度

### WSL 執行指令

```bash
# 影片換臉（v4 優化版）
.venv/bin/python v4/swap_video.py \
    --source images/source.jpg \
    --target input.mp4 \
    --track-interval 5

# 攝影機換臉（v4 優化版，建議 Windows 直跑）
.venv/bin/python v4/swap_camera.py \
    --source images/source.jpg \
    --track-interval 15 \
    --det-size 240

# 啟用多解析度（防止漏臉）
.venv/bin/python v4/swap_video.py \
    --source images/source.jpg \
    --target input.mp4 \
    --track-interval 5 --multi-res
```

### 參數說明

| 參數 | v2/v3 版本 | v4 版本 | 差異 |
|------|-----------|---------|------|
| `--det-size` | 640 (v2) / 320 (v3) | 320 (video) / 240 (camera) | v4 預設更低，因為靠追蹤補償 |
| `--every-nth` | ✅ 有 | ❌ 移除 | 被 `--track-interval` 取代 |
| `--track-interval` | ❌ 無 | ✅ 新增 | 每隔 N 幀才做偵測（預設 5） |
| `--provider` | ❌ 無 | ✅ 新增 | ONNX provider 切換 |
| `--multi-res` | ❌ 無 | ✅ 新增 | 多解析度自動重試 |
| `--enhance` | ✅ 有 | ❌ 移除 | 由外部選擇性實作 |
| `--poisson` | ✅ v3 only | ❌ 移除 | 同上 |

### 實測對比（CPU, 512x512, 50 frames, 10 FPS）

| 版本 | 模式 | 耗時 | 速度 | 偵測次數 |
|------|------|------|------|---------|
| v2 | 每幀偵測 | 110.4s | 0.5 fps | 50/50 (100%) |
| v2 | `--every-nth 3` | 44.1s | 1.1 fps | 17/50 (34%) |
| **v4** | `--track-interval 5` | **73.1s** | **0.7 fps** | **10/50 (20%)** |

> v4 只做 20% 的偵測，節省約 40% 時間。速度低於 `--every-nth` 模式的原因是 v4 每幀仍跑 swap（只是不做 detection），而 v2 的 `--every-nth 3` 連 swap 都跳過了。

---

## 參考資料

- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) — 即時換臉開源專案，架構靈感來源
- [insightface](https://github.com/deepinsight/insightface) — 人臉分析工具包，提供 buffalo_l 及 inswapper 模型
- [OpenCV seamlessClone](https://docs.opencv.org/master/df/da0/group__photo__clone.html) — Poisson blending 官方文件
