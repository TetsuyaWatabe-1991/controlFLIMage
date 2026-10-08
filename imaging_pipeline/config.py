"""Day settings saved beside the acquisition folder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_NAME = "pipeline_config.json"


@dataclass
class PipelineConfig:
    """Settings the window writes. Paths are strings so the file stays plain JSON."""

    pos_csv: str = ""
    savefolder: str = ""
    top_n: int = 8
    min_auto_rating: int = 3
    highmag_setting_path: str = ""
    zoom_highmag: int = 15
    highmag_power_percent: int = 12

    def validate(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.min_auto_rating < 1:
            raise ValueError("min_auto_rating must be at least 1")
        if self.zoom_highmag < 1:
            raise ValueError("zoom_highmag must be at least 1")


def config_path(savefolder: Path) -> Path:
    return savefolder / CONFIG_NAME


def save_config(config: PipelineConfig) -> Path:
    config.validate()
    folder = Path(config.savefolder)
    folder.mkdir(parents=True, exist_ok=True)
    path = config_path(folder)
    path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    return path


def load_config(savefolder: Path) -> PipelineConfig:
    path = config_path(savefolder)
    if not path.is_file():
        return PipelineConfig(savefolder=str(savefolder))
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {field: data[field] for field in PipelineConfig.__dataclass_fields__ if field in data}
    config = PipelineConfig(**known)
    config.validate()
    return config
