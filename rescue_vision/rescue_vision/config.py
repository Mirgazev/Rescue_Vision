from dataclasses import dataclass


@dataclass
class VisionConfig:
    model_path: str = "weights/yolo11_pose_v4.pt"
    source: str = "0"
    enhance_mode: str = "auto"
    imgsz: int = 768
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = "cuda"
    show: bool = False
    save_debug: bool = False
