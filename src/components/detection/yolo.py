import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, List

from utils.logger import LoggerManager
from utils.dtos import Detection, BoundingBox, FrameData
from components.detection.base import BaseDetection


class HumanDetection(BaseDetection):
    """Human Detection using local ONNX Runtime (YOLOv9-c)."""

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.logger: Optional[LoggerManager] = None
        self.img_size = 640
        self.threshold = 0.5
        self.iou_threshold = 0.45

    def initialize(self, config, logger) -> None:
        """Initialize ONNX session."""
        self.logger = logger
        self.camera_id = config.get("camera_id", "unknown")
        self.area_name = config.get("area_name", "unknown_area")
        self.threshold = config.get("threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.img_size = config.get("img_size", 640)
        self.id2label = config.get("class_names", {})
        self.id2label = {int(k): v for k, v in self.id2label.items()}

        onnx_path = config.get("onnx_path", "")
        if not onnx_path:
            raise ValueError("Error config 'onnx_path'")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
            dummy_input = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            _ = self._infer(dummy_input)
            
            self.logger.log_info(
                self.camera_id,
                self.area_name,
                f"Initialize ONNX Human Detection success (path: {onnx_path})",
            )
        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error initializing ONNX Human Detection: {e}",
                exc_info=True,
            )
            raise

    def _letterbox(self, img: np.ndarray):
        shape = img.shape[:2]  # h, w
        r = min(self.img_size / shape[0], self.img_size / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = self.img_size - new_unpad[0], self.img_size - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, r, (dw, dh)

    def _infer(self, preprocessed_img: np.ndarray):
        img_input = preprocessed_img.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)  # HWC -> CHW
        img_input = np.expand_dims(img_input, axis=0)  # Add batch
        
        outputs = self.session.run(None, {'images': img_input})
        return outputs[0]

    def _postprocess(self, pred: np.ndarray, orig_shape: tuple, ratio: float, pad: tuple) -> List[Detection]:
        pred = pred[0]
        pred = pred.transpose(1, 0)

        scores = pred[:, 4:].max(1)
        xc = scores > self.threshold
        pred = pred[xc]

        if len(pred) == 0:
            return []

        boxes = pred[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2  # cx -> x1
        boxes[:, 1] -= boxes[:, 3] / 2  # cy -> y1
        boxes[:, 2] += boxes[:, 0]      # x2
        boxes[:, 3] += boxes[:, 1]      # y2

        scores = pred[:, 4:].max(1)
        classes = pred[:, 4:].argmax(1)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, self.iou_threshold)
        if len(indices) == 0:
            return []

        indices = indices.flatten()
        detections = []
        dw, dh = pad

        for idx in indices:
            box = boxes[idx]
            score = scores[idx]
            cls = int(classes[idx])

            x1 = max(0, int((box[0] - dw) / ratio))
            y1 = max(0, int((box[1] - dh) / ratio))
            x2 = min(orig_shape[1], int((box[2] - dw) / ratio))
            y2 = min(orig_shape[0], int((box[3] - dh) / ratio))

            detection = Detection(
                box=BoundingBox.fromlist([x1, y1, x2, y2]),
                confidence=float(score),
                label=self.id2label.get(cls, "unknown"),
            )
            detections.append(detection)

        return detections

    def execute(self, data: FrameData) -> List[Detection]:
        if not self.session:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                "ONNX session not initialized.",
            )
            return []

        if data is None or data.image is None:
            raise ValueError("FrameData is empty or has no image.")

        orig_img = data.image
        orig_shape = orig_img.shape[:2]

        letterboxed_img, ratio, pad = self._letterbox(orig_img.copy())
        raw_output = self._infer(letterboxed_img)
        detections = self._postprocess(raw_output, orig_shape, ratio, pad)

        return detections