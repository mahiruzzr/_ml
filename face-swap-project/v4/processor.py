import os
import cv2
import numpy as np

os.environ['INSIGHTFACE_LOG_LEVEL'] = 'WARNING'

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


PROVIDER_MAP = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'tensorrt': 'TensorrtExecutionProvider',
    'dml': 'DmlExecutionProvider',
    'openvino': 'OpenVINOExecutionProvider',
}


class FaceTracker:
    def __init__(self, track_interval=10):
        self.track_interval = track_interval
        self.frame_count = 0
        self.cached_faces = []

    def should_detect(self):
        return self.frame_count % self.track_interval == 0

    def tick(self, faces):
        if self.frame_count % self.track_interval == 0:
            self.cached_faces = faces
        self.frame_count += 1
        return self.cached_faces

    def reset(self):
        self.frame_count = 0
        self.cached_faces = []


class OptimizedFaceProcessor:
    def __init__(self, det_size=(320, 320), provider='cpu', track_interval=10, multi_res=False):
        self.det_size = det_size
        self.multi_res = multi_res
        self.provider = PROVIDER_MAP.get(provider, 'CPUExecutionProvider')
        self.providers = [self.provider]

        self.app = FaceAnalysis(
            name='buffalo_l',
            root=os.path.expanduser('~/.insightface'),
            providers=self.providers
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

        model_path = os.path.join(
            os.path.expanduser('~/.insightface'),
            'models',
            'inswapper_128.onnx'
        )
        self.swapper = get_model(model_path, providers=self.providers)

        self.tracker = FaceTracker(track_interval=track_interval)
        self.source_face = None

    def load_source(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        det_sizes = [(320, 320), (640, 640), (192, 192)] if self.multi_res else [self.det_size]
        for ds in det_sizes:
            if ds != self.app.det_size:
                self.app.prepare(ctx_id=0, det_size=ds)
            faces = self.app.get(img_rgb)
            if len(faces) > 0:
                break

        if len(faces) == 0:
            raise ValueError("No face detected in source image.")
        self.source_face = faces[0]
        return self.source_face

    def detect_faces(self, img_rgb):
        if self.tracker.should_detect():
            if self.multi_res:
                sizes = [(320, 320), (640, 640), (192, 192)]
                all_faces = []
                for ds in sizes:
                    if ds != self.app.det_size:
                        self.app.prepare(ctx_id=0, det_size=ds)
                    faces = self.app.get(img_rgb)
                    if len(faces) > len(all_faces):
                        all_faces = faces
                    if len(all_faces) > 0:
                        break
                detected = all_faces
            else:
                if self.det_size != self.app.det_size:
                    self.app.prepare(ctx_id=0, det_size=self.det_size)
                detected = self.app.get(img_rgb)
            return self.tracker.tick(detected)
        else:
            return self.tracker.tick([])

    def swap(self, frame_rgb, target_faces=None):
        if target_faces is None or len(target_faces) == 0:
            return frame_rgb
        result = frame_rgb.copy()
        for face in target_faces:
            result = self.swapper.get(result, face, self.source_face, paste_back=True)
        return result

    def process_frame(self, frame_bgr):
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        frame_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces(frame_rgb)
        result_rgb = self.swap(frame_rgb, faces)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR), len(faces) > 0
