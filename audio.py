"""音频预处理：引擎只认 WAV（内部用 dr_wav 解码），所以其余格式一律先用 ffmpeg 转。

引擎内部会自己做「多声道混单声道 + 线性重采样到 16 kHz」，所以这里只要保证产出是
WAV 即可；不过统一转成 16k 单声道 pcm_s16le 能顺带把体积压下来，也避免 dr_wav 碰到
一些冷门的 WAV 变体。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 转码产物放 tmp/（每次启动会清空），转写结果放 out/（保留，供重复下载）
WORK_DIR = ROOT / "runtime" / "tmp"

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class AudioError(Exception):
    """音频探测/转码失败，错误信息可直接在界面上展示。"""


def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


def require_ffmpeg() -> None:
    if not FFMPEG or not FFPROBE:
        raise AudioError(
            "系统里找不到 ffmpeg / ffprobe。引擎只认 WAV，非 WAV 音频必须靠 ffmpeg 转码。\n"
            "请先安装（例如 sudo dnf install ffmpeg 或 sudo apt install ffmpeg）。"
        )


def probe(path: str | Path) -> dict:
    """返回时长（秒）与音频流编码名。"""
    require_ffmpeg()
    p = _run([
        FFPROBE, "-v", "error", "-show_entries",
        "format=duration:stream=codec_name",
        "-select_streams", "a:0", "-of", "json", str(path),
    ])
    if p.returncode != 0:
        raise AudioError(
            "无法读取这个音频文件，可能已损坏或不是有效的音视频容器。\n"
            f"ffprobe 报错：{(p.stderr or '').strip()[:500]}"
        )
    try:
        info = json.loads(p.stdout)
        duration = float(info["format"]["duration"])
        codec = (info.get("streams") or [{}])[0].get("codec_name", "")
    except Exception:
        raise AudioError("解析 ffprobe 输出失败，文件可能没有音轨。")
    if duration <= 0:
        raise AudioError("探测到的音频时长为 0，文件里可能没有音轨。")
    return {"duration": duration, "codec": codec}


def ensure_wav(src: str | Path, max_minutes: float) -> tuple[Path, float]:
    """把任意格式的音频变成引擎能吃的 WAV，返回 (wav 路径, 时长秒)。

    始终产出自己的副本，避免直接引用 Gradio 的临时文件（它随时可能被清理）。
    """
    require_ffmpeg()
    src = Path(src)
    if not src.is_file():
        raise AudioError(f"音频文件不存在：{src}")

    info = probe(src)
    duration = info["duration"]
    minutes = duration / 60.0
    if max_minutes and max_minutes > 0 and minutes > max_minutes:
        raise AudioError(
            f"音频时长 {minutes:.1f} 分钟，超过设定的上限 {max_minutes:.0f} 分钟。\n"
            "这个模型是自回归解码，RTF 会随音频变长而上升（132 秒音频在 8 线程 CPU 上约需 100 秒），"
            "过长的音频会跑很久。确认要跑的话请把「时长上限」调大。"
        )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dst = WORK_DIR / (src.stem[:60] + ".wav")

    # 已经是 PCM WAV 就直接复制，省一次重编码。
    if src.suffix.lower() == ".wav" and info["codec"].startswith("pcm_"):
        shutil.copyfile(src, dst)
    else:
        p = _run([
            FFMPEG, "-y", "-nostdin", "-i", str(src),
            "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(dst),
        ], timeout=max(120, int(duration / 3)))
        if p.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
            raise AudioError(
                "ffmpeg 转码失败，没能生成可用的 WAV。\n"
                f"ffmpeg 报错：{(p.stderr or '').strip()[-800:]}"
            )
    return dst, duration


def split_wav(src: str | Path, segment_seconds: float = 60.0) -> list[Path]:
    """把 WAV 按固定时长切成多段，返回每段 WAV 路径列表（按文件名排序）。

    用 ffmpeg segment 复用音轨（16k 单声道 pcm_s16le），不损失质量；每段时间戳重置为 0，
    因为分段模式下时间戳本就不可靠，输出会丢弃时间标注。切片产物落在 runtime/tmp/，
    下次启动会被 cleanup() 清掉。
    """
    require_ffmpeg()
    src = Path(src)
    if not src.is_file():
        raise AudioError(f"要切片的 WAV 不存在：{src}")
    try:
        seg = float(segment_seconds)
    except (TypeError, ValueError):
        seg = 60.0
    if seg < 10:
        seg = 60.0

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    base = src.stem[:60]
    out_pattern = WORK_DIR / (base + "_seg_%03d.wav")
    p = _run([
        FFMPEG, "-y", "-nostdin", "-i", str(src),
        "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", f"{seg:.3f}", "-reset_timestamps", "1",
        str(out_pattern),
    ], timeout=300)
    if p.returncode != 0:
        raise AudioError(
            "ffmpeg 切片失败，没能切出可用的分段。\n"
            f"ffmpeg 报错：{(p.stderr or '').strip()[-800:]}"
        )
    parts = sorted(WORK_DIR.glob(base + "_seg_*.wav"))
    if not parts:
        raise AudioError("ffmpeg 切片没有产出任何文件。")
    return parts


def cleanup() -> None:
    """启动时清掉上一轮遗留的转码中间产物（不影响 out/ 里的转写结果）。"""
    if WORK_DIR.is_dir():
        shutil.rmtree(WORK_DIR, ignore_errors=True)
