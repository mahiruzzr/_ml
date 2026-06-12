import argparse
import cv2
import os
import sys
import time

from processor import OptimizedFaceProcessor


def main():
    parser = argparse.ArgumentParser(description='V4: Optimized Video Face Swap')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', default='output/result_v4.mp4')
    parser.add_argument('--det-size', type=int, default=320)
    parser.add_argument('--provider', default='cpu', choices=['cpu', 'cuda', 'dml', 'openvino'])
    parser.add_argument('--track-interval', type=int, default=5,
                        help='Run face detection every N frames (default: 5)')
    parser.add_argument('--multi-res', action='store_true',
                        help='Try multiple resolutions if no face found')
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

    print(f'[V4] Opening video: {args.target}')
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        print(f'ERROR: Cannot open {args.target}')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  {width}x{height}, {fps:.2f} FPS, {total} frames')

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    start = time.time()
    detect_count = 0

    print('[V4] Processing...')
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        result_bgr, face_found = proc.process_frame(frame_bgr)
        out.write(result_bgr)
        frame_idx += 1
        if proc.tracker.should_detect():
            detect_count += 1

        if frame_idx % max(1, total // 20) == 0 or frame_idx == total:
            elapsed = time.time() - start
            print(f'  [{frame_idx/total*100:5.1f}%] {frame_idx}/{total} | '
                  f'{frame_idx/elapsed:.1f} fps | detection runs: {detect_count}/{frame_idx}')

    cap.release()
    out.release()
    elapsed = time.time() - start
    print(f'[V4] Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)')
    print(f'  Face detections: {detect_count}/{frame_idx} '
          f'(skipped {frame_idx - detect_count} frames, {100*(frame_idx-detect_count)/max(1,frame_idx):.0f}%)')
    print(f'  Output: {args.output}')


if __name__ == '__main__':
    main()
