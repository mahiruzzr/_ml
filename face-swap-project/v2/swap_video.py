import argparse
import cv2
import numpy as np
import os
import sys
import time

os.environ['INSIGHTFACE_LOG_LEVEL'] = 'WARNING'

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


def init_face_analyser(det_size=(640, 640)):
    app = FaceAnalysis(
        name='buffalo_l',
        root=os.path.expanduser('~/.insightface'),
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def init_face_swapper():
    model_path = os.path.join(
        os.path.expanduser('~/.insightface'),
        'models',
        'inswapper_128.onnx'
    )
    swapper = get_model(model_path, providers=['CPUExecutionProvider'])
    return swapper


def load_source_face(app, path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    if len(faces) == 0:
        print("ERROR: No face detected in source image.")
        sys.exit(1)
    return faces[0]


def enhance_face(result_rgb, original_bgr, face_bbox, strength=0.5):
    h, w = result_rgb.shape[:2]
    x1, y1, x2, y2 = face_bbox[:4]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    face_w, face_h = x2 - x1, y2 - y1
    if face_w < 5 or face_h < 5:
        return result_rgb

    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    result_face = result_bgr[y1:y2, x1:x2]
    orig_face = original_bgr[y1:y2, x1:x2]

    result_lab = cv2.cvtColor(result_face, cv2.COLOR_BGR2LAB)
    orig_lab = cv2.cvtColor(orig_face, cv2.COLOR_BGR2LAB)

    for c in range(3):
        r_ch = result_lab[:, :, c]
        r_mean, r_std = r_ch.mean(), r_ch.std()
        o_mean, o_std = orig_lab[:, :, c].mean(), orig_lab[:, :, c].std()

        if r_std > 0:
            r_ch = ((r_ch - r_mean) * (o_std / r_std) + o_mean).clip(0, 255).astype(np.uint8)
        r_ch = (r_ch.astype(np.float32) * (1 - strength) + orig_lab[:, :, c].astype(np.float32) * strength).clip(0, 255).astype(np.uint8)
        result_lab[:, :, c] = r_ch

    blended_face = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    canvas = result_bgr.copy()
    canvas[y1:y2, x1:x2] = blended_face

    kernel_size = max(3, min(face_w, face_h) // 10) | 1
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = face_w // 2, face_h // 2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    mask_3ch = np.stack([mask] * 3, axis=2)
    final_bgr = (result_bgr * (1 - mask_3ch) + canvas * mask_3ch).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)


def process_frame(swapper, app, frame_rgb, source_face, original_bgr=None, enhance=False):
    faces = app.get(frame_rgb)
    result = frame_rgb.copy()
    for face in faces:
        result = swapper.get(result, face, source_face, paste_back=True)
        if enhance and original_bgr is not None:
            bbox = face.bbox.astype(int)
            result = enhance_face(result, original_bgr, bbox)
    return result


def main():
    parser = argparse.ArgumentParser(description='V2: Video Face Swap')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target', required=True, help='Target video file path')
    parser.add_argument('--output', default='output/result.mp4', help='Output video path')
    parser.add_argument('--det-size', type=int, default=640, help='Detection resolution (default: 640)')
    parser.add_argument('--every-nth', type=int, default=1, help='Process every Nth frame, copy rest from previous result (default: 1)')
    parser.add_argument('--enhance', action='store_true', help='Enable face color enhancement')
    args = parser.parse_args()

    print('[V2] Initializing face analyser...')
    app = init_face_analyser(det_size=(args.det_size, args.det_size))

    print('[V2] Initializing face swapper...')
    swapper = init_face_swapper()

    print(f'[V2] Loading source face: {args.source}')
    source_face = load_source_face(app, args.source)
    print(f'  Source face detected (score={source_face.det_score:.3f})')

    print(f'[V2] Opening target video: {args.target}')
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        print(f'ERROR: Cannot open video: {args.target}')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  {width}x{height}, {fps:.2f} FPS, {total_frames} frames')

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    cached_result = None
    frame_idx = 0
    start_time = time.time()

    print('[V2] Processing frames...')
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % args.every_nth == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result_rgb = process_frame(swapper, app, frame_rgb, source_face, frame_bgr if args.enhance else None, args.enhance)
            cached_result = result_rgb
        elif cached_result is not None:
            result_rgb = cached_result
        else:
            result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)

        frame_idx += 1

        if frame_idx % max(1, total_frames // 20) == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            pct = frame_idx / total_frames * 100
            print(f'  [{pct:5.1f}%] frame {frame_idx}/{total_frames} | {frame_idx/elapsed:.1f} fps' if elapsed > 0 else f'  [{pct:5.1f}%] frame {frame_idx}/{total_frames}')

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f'[V2] Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)')
    print(f'  Output: {args.output}')


if __name__ == '__main__':
    main()
