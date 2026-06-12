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


def init_face_analyser(det_size=(320, 320)):
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


def poisson_blend(swapped_bgr, original_bgr, face_bbox):
    h, w = swapped_bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox[:4]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    face_w, face_h = x2 - x1, y2 - y1
    if face_w < 5 or face_h < 5:
        return swapped_bgr

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = face_w // 2, face_h // 2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    result = cv2.seamlessClone(swapped_bgr, original_bgr, mask, (cx, cy), cv2.NORMAL_CLONE)
    return result


def draw_overlay(img, lines):
    for i, line in enumerate(lines):
        y = 25 + i * 25
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def main():
    parser = argparse.ArgumentParser(description='V3: Real-time Camera Face Swap')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--camera', type=int, default=None, help='Camera device ID (e.g. 0)')
    parser.add_argument('--video', type=str, default=None, help='Video file path (alternative to camera)')
    parser.add_argument('--det-size', type=int, default=320, help='Detection resolution (default: 320)')
    parser.add_argument('--every-nth', type=int, default=3, help='Process every Nth frame (default: 3)')
    parser.add_argument('--display-scale', type=float, default=1.0)
    parser.add_argument('--poisson', action='store_true', help='Enable Poisson blending')
    args = parser.parse_args()

    print('[V3] Initializing models...')
    app = init_face_analyser(det_size=(args.det_size, args.det_size))
    swapper = init_face_swapper()

    print(f'[V3] Loading source face: {args.source}')
    source_face = load_source_face(app, args.source)
    print(f'  Source face score={source_face.det_score:.3f}')

    cap = None
    is_video = args.video is not None

    if is_video:
        print(f'[V3] Opening video: {args.video}')
        cap = cv2.VideoCapture(args.video)
    elif args.camera is not None:
        print(f'[V3] Opening camera #{args.camera}...')
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print('ERROR: specify --camera N or --video path')
        sys.exit(1)

    if not cap.isOpened():
        print(f'ERROR: Cannot open input')
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_type = 'Video' if is_video else 'Camera'
    print(f'  {src_type}: {width}x{height}' + (f', {total} frames' if is_video else ''))

    cv2.namedWindow('Face Swap', cv2.WINDOW_NORMAL)
    print()
    print('  Click on the window to give it focus, then:')
    print('    Q - quit')
    print()

    frame_idx = 0
    cached_result = None
    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if not is_video and frame_idx < 30:
            show = frame_bgr.copy()
            draw_overlay(show, ['Starting...', 'Press Q to quit'])
            cv2.imshow('Face Swap', show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1
            continue

        processed = False
        if frame_idx % args.every_nth == 0:
            enhance_bgr = frame_bgr.copy()
            lab = cv2.cvtColor(enhance_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhance_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            frame_rgb = cv2.cvtColor(enhance_bgr, cv2.COLOR_BGR2RGB)
            faces = app.get(frame_rgb)
            if len(faces) > 0:
                result_rgb = frame_rgb.copy()
                for face in faces:
                    result_rgb = swapper.get(result_rgb, face, source_face, paste_back=True)
                if args.poisson:
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        result_bgr = poisson_blend(result_bgr, frame_bgr, bbox)
                    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                cached_result = result_rgb
                processed = True
            else:
                cached_result = None

        if processed:
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        elif cached_result is not None:
            result_bgr = cv2.cvtColor(cached_result, cv2.COLOR_RGB2BGR)
        else:
            result_bgr = frame_bgr.copy()
            cv2.putText(result_bgr, 'No face detected', (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frame_idx += 1
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0

        draw_overlay(result_bgr, [
            f'V3 | {fps:.0f} FPS | every-nth={args.every_nth}',
            f'Poisson: {"ON" if args.poisson else "OFF"} | Q = quit',
        ])

        if args.display_scale != 1.0:
            dw = int(result_bgr.shape[1] * args.display_scale)
            dh = int(result_bgr.shape[0] * args.display_scale)
            result_bgr = cv2.resize(result_bgr, (dw, dh))

        cv2.imshow('Face Swap', result_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    label = 'Video' if is_video else 'Camera'
    print(f'[V3] {label} stopped. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} avg FPS)')


if __name__ == '__main__':
    main()
