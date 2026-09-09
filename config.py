"""界面设置的持久化：HSA_OVERRIDE_GFX_VERSION 等值写进 JSON，下次启动沿用。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "mossweb.settings.json"

DEFAULTS = {
    "backend": "auto",          # auto | rocm | vulkan
    "hsa_override_gfx_version": "1150",
    "mtd_device": "auto",       # auto | cpu | gpu
    "mtd_threads": 8,
    "output_format": "srt",     # text | srt | ass | json
    "max_new": -1,              # <=0 表示用 GGUF 里的默认值
    "max_audio_minutes": 60,
    "timeout_minutes": 120,
    "segmented": False,            # 分段处理：按固定时长切片，逐段转写后合并
    "segment_seconds": 60,         # 切片时长（秒），默认每分钟一段
    "last_model": "",
}


def load() -> dict:
    data = dict(DEFAULTS)
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k in DEFAULTS:
                    data[k] = v
    except FileNotFoundError:
        pass
    except Exception:
        # 配置文件损坏时宁可用默认值，也不要让界面起不来
        pass
    return data


def save(data: dict) -> None:
    """先写临时文件再原子替换，避免写到一半被打断导致配置损坏。"""
    payload = {k: data.get(k, DEFAULTS[k]) for k in DEFAULTS}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
