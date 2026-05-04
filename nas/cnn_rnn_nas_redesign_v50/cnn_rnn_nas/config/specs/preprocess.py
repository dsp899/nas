from dataclasses import dataclass

from ..supported.shared_supported import RESIZE_MODES, SAMPLING_MODES


@dataclass(frozen=True)
class FrameSourceSpec:
    image_size: int
    frames: int
    sampling: str
    resize_mode: str

    def __post_init__(self) -> None:
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode no soportado: {self.resize_mode!r}")
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling no soportado: {self.sampling!r}")
        if int(self.image_size) <= 0:
            raise ValueError("image_size debe ser > 0")
        if int(self.frames) <= 0:
            raise ValueError("frames debe ser > 0")

    @property
    def tag(self) -> str:
        return f"preprocess_{int(self.frames)}_{self.sampling}_frames"


@dataclass(frozen=True)
class PreprocessSpec:
    image_size: int
    train_frames: int
    predict_frames: int
    sampling: str
    resize_mode: str
    shuffle_buffer_videos: int = 1024
    shuffle_buffer_frames: int = 5000

    def __post_init__(self) -> None:
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode no soportado: {self.resize_mode!r}")
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling no soportado: {self.sampling!r}")
        if int(self.image_size) <= 0:
            raise ValueError("image_size debe ser > 0")
        if int(self.train_frames) <= 0 or int(self.predict_frames) <= 0:
            raise ValueError("train_frames y predict_frames deben ser > 0")

    @property
    def train_source(self) -> FrameSourceSpec:
        return FrameSourceSpec(image_size=self.image_size, frames=self.train_frames, sampling=self.sampling, resize_mode=self.resize_mode)

    @property
    def test_source(self) -> FrameSourceSpec:
        return self.train_source

    @property
    def predict_source(self) -> FrameSourceSpec:
        return FrameSourceSpec(image_size=self.image_size, frames=self.predict_frames, sampling=self.sampling, resize_mode=self.resize_mode)

    @property
    def train_tag(self) -> str:
        return self.train_source.tag

    @property
    def test_tag(self) -> str:
        return self.test_source.tag

    @property
    def predict_tag(self) -> str:
        return self.predict_source.tag


@dataclass(frozen=True)
class AugmentationSpec:
    enabled: bool = False
    random_flip: bool = False
    random_crop_scale_min: float = 0.85
    brightness_delta: float = 0.15
    contrast_lower: float = 0.85
    contrast_upper: float = 1.15
    saturation_lower: float = 0.85
    saturation_upper: float = 1.15

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.random_crop_scale_min) <= 1.0:
            raise ValueError("random_crop_scale_min debe estar en [0, 1]")
        if not 0.0 <= float(self.brightness_delta) <= 1.0:
            raise ValueError("brightness_delta debe estar en [0, 1]")
