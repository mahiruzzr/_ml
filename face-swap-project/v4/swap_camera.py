import argparse
import cv2
import os
import sys
import time

from processor import OptimizedFaceProcessor


def main():
    parser = argparse.ArgumentParser(description='V4: Optimized Camera Face Swap')
    parser.add_argument('--source', required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--det-size', type=int, default=240)
    parser.add_argument('--provider', default='cpu', choices=['cpu', 'cuda', 'dml', 'openvino'])
    parser.add_argument('--track-interval', type=int, default=15,
                        help='Run face detection every N frames (default: 15)')
    parser.add_argument('--display-scale', type=float, default=1.0)
    parser.add_argument('--multi-res', action='store_true')
    args = parser.parse_args()

    print(f'[V4] Initializing processor (provider={args.provider}, track_interval={args.track_interval})')
    proc = OptimizedFaceProcessor(
        det_size=(args.det_size, args.det_size),
        provider=args.provider,
        track_interval=args.track_interval,
        multi_res=args.multi_res,
    )

    print(f'[V4] Loading source face: {args.source}')
    proc.load_source(args.source)
    print(f'  Source: score={proc.source_face.det_score:.3f}')

    print(f'[V4] Opening camera #{args.camera}...')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera #{args.camera}')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  Camera: {width}x{height}')

    cv2.namedWindow('Face Swap (V4)', cv2.WINDOW_NORMAL)
    print()
    print('  Click on the window to give it focus, then:')
    print('    Q - quit')
    print()

    frame_idx = 0
    start = time.time()
    detect_count = 0
    warmup = True

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if warmup:
            show = frame_bgr.copy()
            for i, line in enumerate(['V4 starting...', 'Press Q to quit']):
                cv2.putText(show, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Face Swap (V4)', show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_idx > 30:
                warmup = False
            frame_idx += 1
            continue

        result_bgr, face_found = proc.process_frame(frame_bgr)
        frame_idx += 1
        if proc.tracker.should_detect():
            detect_count += 1

        if not face_found:
            cv2.putText(result_bgr, 'No face detected', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        elapsed = time.time() - start
        display_fps = frame_idx / elapsed if elapsed > 0 else 0

        lines = [
            f'V4 | {display_fps:.0f} FPS | track_interval={args.track_interval}',
            f'Detection runs: {detect_count}/{frame_idx} | Q = quit',
        ]
        for i, line in enumerate(lines):
            cv2.putText(result_bgr, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(result_bgr, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        if args.display_scale != 1.0:
            dw = int(result_bgr.shape[1] * args.display_scale)
            dh = int(result_bgr.shape[0] * args.display_scale)
            result_bgr = cv2.resize(result_bgr, (dw, dh))

        cv2.imshow('Face Swap (V4)', result_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start
    print(f'[V4] Stopped. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} avg FPS)')


if __name__ == '__main__':
    main()
