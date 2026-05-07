from dataclasses import dataclass
from pathlib import Path


Size = tuple[int, int]
Roi = tuple[int, int, int, int]


@dataclass
class SlicerConfig:
    video_folder: Path
    output_folder: Path
    max_screenshots_per_video: int = 300
    target_size: Size | None = None
    auto_enabled: bool = True
    auto_step: int = 1
    use_roi: bool = False
    overwrite: bool = False

    @property
    def resize_enabled(self) -> bool:
        return self.target_size is not None
