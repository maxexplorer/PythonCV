from dataclasses import dataclass, field
from pathlib import Path


Size = tuple[int, int]
Roi = tuple[int, int, int, int]


@dataclass
class SlicerConfig:
    video_folder: Path
    output_folder: Path
    selected_video_files: list[Path] = field(default_factory=list)
    target_size: Size | None = None
    auto_enabled: bool = True
    auto_step: int = 1
    use_roi: bool = False

    @property
    def resize_enabled(self) -> bool:
        return self.target_size is not None
