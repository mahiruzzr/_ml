import argparse
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insightface_model

# ==============================================================================
# 第一步：初始化 FaceAnalysis（這是最重要的前置作業）
# ==============================================================================
def init_face_analyser():
    """
    初始化 insightface 的 FaceAnalysis。

    為什麼用 FaceAnalysis 而不是直接呼叫 model？
    - FaceAnalysis 是一個「整合管理器」，它會自動載入三個子模型：
      ① RetinaFace（人臉偵測）— 找到圖片中的人臉 bounding box
      ② ArcFace（人臉辨識）— 把人臉轉成 512 維的特徵向量（embedding）
      ③  Landmark 2D 106（臉部關鍵點）— 106 個臉部特徵點，用於對齊

    為什麼用 buffalo_l？
    - insightface 提供多個模型包：buffalo_m、buffalo_l、buffalo_s
    - l = large，準確率最高但速度稍慢（v1 我們要品質，速度等 v4 再優化）
    - m = medium，s = small（更快但較不準）

    為什麼 det_size=(640, 640)？
    - RetinaFace 的輸入解析度，640x640 是準確率跟速度的平衡點
    - 越大越準但越慢，越小越快但容易漏掉小臉
    """
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


# ==============================================================================
# 第二步：初始化換臉模型
# ==============================================================================
def init_face_swapper():
    model_path = os.path.join(
        os.path.expanduser('~/.insightface'),
        'models',
        'inswapper_128.onnx'
    )
    swapper = get_insightface_model(
        model_path,
        providers=['CPUExecutionProvider']
    )
    return swapper


# ==============================================================================
# 第三步：載入圖片 + 前處理
# ==============================================================================
def load_image(path: str) -> np.ndarray:
    """
    用 OpenCV 載入圖片。

    為什麼用 OpenCV？（為什麼不是 PIL？）
    - insightface 底層吃的是 OpenCV 的 BGR 格式（不是 RGB！）
    - OpenCV 的 numpy array（H×W×C）是大多數 CV pipeline 的標準格式
    - 後續的 affine transform、resize 都可以直接用 cv2 函式

    什麼是 BGR vs RGB？
    - OpenCV 載入圖片時預設是 BGR（藍-綠-紅）順序
    - 一般圖像格式是 RGB（紅-綠-藍）
    - 如果你用 PIL 載入然後餵給 insightface，顏色會爆炸
    - 所以統一用 cv2.imread() 最安全
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


# ==============================================================================
# 第四步：偵測人臉 + 提取特徵
# ==============================================================================
def detect_faces(app: FaceAnalysis, img: np.ndarray):
    """
    用 FaceAnalysis 偵測圖片中的所有臉孔。

    app.get(img) 回傳一個 list of Face 物件，每個 Face 包含：
    - bbox：臉部邊界框 [x1, y1, x2, y2, confidence]
    - kps：5 個臉部關鍵點（左眼、右眼、鼻子、左嘴角、右嘴角）
    - det_score：偵測信心分數
    - embedding：512 維的特徵向量（就是 ArcFace 的輸出！）

    如果回傳空 list → 沒有偵測到人臉
    如果 len > 1 → 有多張臉，但 v1 只處理第一張
    （多張臉的處理等 v3 再實作）
    """
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError(f"No face detected in image")
    return faces


# ==============================================================================
# 第五步：執行換臉
# ==============================================================================
def swap_face(swapper, source_img: np.ndarray, target_img: np.ndarray,
              source_face, target_face) -> np.ndarray:
    """
    真正的換臉核心邏輯。

    參數：
    - source_img：source 的完整圖片（用來貼臉部特徵）
    - target_img：target 的完整圖片（被換臉的那張）
    - source_face：source 的人臉物件（包含 embedding）
    - target_face：target 的人臉物件（包含位置、關鍵點）

    swapper.get() 內部做了什麼？
    ① 根據 target_face 的關鍵點（kps）計算 affine transform
    ② 把 target 人臉 crop 出來並 align 到 128×128
    ③ 把 source 的 embedding 跟 target 的 aligned face 一起餵進模型
    ④ 模型生成新的臉部圖像（source 的身分 + target 的姿勢）
    ⑤ 用 inverse affine transform 把生成結果貼回 target 原圖位置
    ⑥ 回傳修改後的完整圖片

    為什麼要傳 source_img 和 target_img？
    - swapper.get() 需要原始圖片來做 affine transform 的取樣
    - 它不像 Deep-Live-Cam 那樣另外做 Poisson blending（那是 v2 的事）
    """
    result = swapper.get(target_img, target_face, source_face, paste_back=True)
    return result


# ==============================================================================
# 第六步：後處理 + 輸出
# ==============================================================================
def save_image(path: str, img: np.ndarray):
    """
    儲存處理後的圖片。

    cv2.imwrite() 吃 BGR 格式，跟我們整條 pipeline 一致。
    如果副檔名是 .jpg，品質參數可以調（預設 95）。
    """
    success = cv2.imwrite(path, img)
    if not success:
        raise IOError(f"Failed to save image: {path}")


# ==============================================================================
# 主程式
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='V1: Static Image Face Swap')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target', required=True, help='Target image path (to be swapped)')
    parser.add_argument('--output', default='output/result.jpg', help='Output image path')
    args = parser.parse_args()

    # Step 1: Initialize models
    print('[1/5] Initializing FaceAnalysis (RetinaFace + ArcFace)...')
    app = init_face_analyser()

    print('[2/5] Initializing face swapper (inswapper_128)...')
    swapper = init_face_swapper()

    # Step 2: Load images
    print('[3/5] Loading images...')
    source_img = load_image(args.source)
    target_img = load_image(args.target)

    # Step 3: Detect faces
    print('[4/5] Detecting faces...')
    source_faces = detect_faces(app, source_img)
    target_faces = detect_faces(app, target_img)
    print(f'  -> Source: {len(source_faces)} face(s)')
    print(f'  -> Target: {len(target_faces)} face(s)')

    # Step 4: Execute face swap
    source_face = source_faces[0]
    target_face = target_faces[0]

    print('[5/5] Swapping face...')
    result = swap_face(swapper, source_img, target_img, source_face, target_face)

    # Step 5: Save result
    save_image(args.output, result)
    print(f'[OK] Face swap complete! Output: {args.output}')


if __name__ == '__main__':
    main()
