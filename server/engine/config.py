"""One place for tunable engine settings."""
from dataclasses import dataclass


@dataclass
class EngineConfig:
    model_size: str = "n"          # n / s / m / l / x
    img_size: int = 640            # network input resolution
    device: str = "cuda"           # "cuda" or "cpu"
    conf_threshold: float = 0.25   # detection confidence cutoff
    iou_threshold: float = 0.45    # NMS IoU cutoff
    half: bool = True              # FP16 inference on GPU
    max_detections: int = 300      # cap per frame
    detect_interval: int = 1       # run the detector every N frames
    weights: str | None = None     # explicit weights path (overrides model_size)

    @property
    def detect_weights(self) -> str:
        # An explicit path (e.g. a trained best.pt) wins; otherwise pick
        # the pretrained model for the configured size.
        return self.weights or f"yolo26{self.model_size}.pt"
