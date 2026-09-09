"""moss-transcribe.cpp 命令行封装。

核心是编译期绑定后端的：core/rocm/ 只能跑 HIP/ROCm，core/vulkan/ 只能跑 Vulkan。
所以「切换后端」= 切换使用哪个目录下的二进制，并让 LD_LIBRARY_PATH 指过去
（两个 CLI 的 ELF RUNPATH 都指向原仓库的 build-hip/ build-vk/，不覆盖的话会去加载
那边的 .so，脱离工作目录就跑不了）。

ROCm 的 HSA_OVERRIDE_GFX_VERSION 必须在进程启动前、也就是 dlopen 之前注入，
用 subprocess 的 env= 天然满足这一点。
"""

from __future__ import annotations

import os
import json
import queue
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parent
CORE_DIR = ROOT / "core"

BACKEND_DIRS = {
    "rocm": CORE_DIR / "rocm",
    "vulkan": CORE_DIR / "vulkan",
}

# 检测到使用 ROCm 后端时，默认自动导出的 GFX 版本覆盖值
DEFAULT_HSA_OVERRIDE_GFX_VERSION = "1150"

OUTPUT_FORMATS = ("text", "srt", "ass", "json")
FORMAT_SUFFIX = {"text": ".txt", "srt": ".srt", "ass": ".ass", "json": ".json"}

# 引擎日志里形如 "[mt I] backend: CPU"
_DEVICE_RE = re.compile(r"backend:\s*(\S+)")

_current: dict = {"proc": None}


class EngineError(Exception):
    """命令还没跑起来就发现的问题。"""


@dataclass
class RunRequest:
    model: str
    wav: str
    backend: str = "auto"
    output_format: str = "srt"
    max_new: int = -1
    hsa_override_gfx_version: str = DEFAULT_HSA_OVERRIDE_GFX_VERSION
    mtd_device: str = "auto"      # auto | cpu | gpu
    mtd_threads: int = 8
    timeout: float = 0.0          # 秒，<=0 表示不限


@dataclass
class RunResult:
    ok: bool
    text: str = ""
    log: str = ""
    exit_code: Optional[int] = None
    error: str = ""
    elapsed: float = 0.0
    cmd: list = field(default_factory=list)
    backend: str = ""
    binary: str = ""
    device: str = ""


# --------------------------------------------------------------------------
# 环境探测
# --------------------------------------------------------------------------

def backend_status(name: str) -> tuple[bool, str]:
    d = BACKEND_DIRS.get(name)
    if d is None:
        return False, f"未知后端：{name}"
    exe = d / "moss-transcribe"
    if not exe.is_file():
        return False, f"缺少可执行文件：{exe}"
    if not os.access(exe, os.X_OK):
        return False, f"文件不可执行（试试 chmod +x）：{exe}"
    if not (d / "libmoss-transcribe.so").is_file():
        return False, f"缺少动态库：{d / 'libmoss-transcribe.so'}"
    return True, ""


def rocm_runtime_present() -> bool:
    """ROCm 用户态库装了不等于能用，还得有 /dev/kfd 和 /dev/dri。"""
    return os.path.exists("/dev/kfd") and os.path.exists("/dev/dri")


def resolve_backend(pref: str) -> tuple[str, str]:
    """返回 (后端名, 备注)。pref 为 auto 时按运行时可用性挑。"""
    if pref in BACKEND_DIRS:
        ok, why = backend_status(pref)
        if not ok:
            raise EngineError(why)
        return pref, ""

    rocm_ok, _ = backend_status("rocm")
    vulkan_ok, _ = backend_status("vulkan")

    if rocm_ok and rocm_runtime_present():
        return "rocm", ""
    if vulkan_ok:
        note = "未检测到 /dev/kfd，ROCm 运行时不可用，已改用 Vulkan" if rocm_ok else ""
        return "vulkan", note
    if rocm_ok:
        return "rocm", "警告：未检测到 /dev/kfd 与 /dev/dri，ROCm 大概率会初始化失败"
    raise EngineError(
        "core/rocm 与 core/vulkan 下都没有可用的核心，请检查工作目录结构。"
    )


def backend_summary() -> str:
    lines = []
    for name in ("rocm", "vulkan"):
        ok, why = backend_status(name)
        mark = "可用" if ok else "不可用"
        lines.append(f"- **{name}**：{mark}" + (f"（{why}）" if why else ""))
    lines.append(
        f"- ROCm 运行时设备节点：{'已检测到 /dev/kfd 与 /dev/dri' if rocm_runtime_present() else '未检测到 /dev/kfd 或 /dev/dri'}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 命令与环境变量
# --------------------------------------------------------------------------

def build_cmd(req: RunRequest, binary: Path) -> list[str]:
    cmd = [str(binary), "transcribe", str(req.model), str(req.wav),
           "--format", req.output_format]
    try:
        max_new = int(req.max_new)
    except (TypeError, ValueError):
        max_new = -1
    if max_new > 0:
        cmd += ["--max-new", str(max_new)]
    return cmd


def build_env(req: RunRequest, backend: str) -> dict:
    env = os.environ.copy()
    backend_dir = str(BACKEND_DIRS[backend])

    # RUNPATH 优先级低于 LD_LIBRARY_PATH，所以这么设能确保用的是工作目录下的 .so
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = backend_dir + (os.pathsep + prev if prev else "")

    if backend == "rocm":
        env["HSA_OVERRIDE_GFX_VERSION"] = str(
            req.hsa_override_gfx_version or DEFAULT_HSA_OVERRIDE_GFX_VERSION
        ).strip() or DEFAULT_HSA_OVERRIDE_GFX_VERSION
    else:
        # Vulkan 不需要它，留着反而可能干扰
        env.pop("HSA_OVERRIDE_GFX_VERSION", None)

    device = (req.mtd_device or "auto").strip().lower()
    if device and device != "auto":
        env["MTD_DEVICE"] = device
    else:
        env.pop("MTD_DEVICE", None)

    try:
        threads = int(req.mtd_threads)
    except (TypeError, ValueError):
        threads = 0
    if threads > 0:
        env["MTD_THREADS"] = str(threads)
    else:
        env.pop("MTD_THREADS", None)

    return env


# --------------------------------------------------------------------------
# 执行（流式 + 可取消）
# --------------------------------------------------------------------------

def _pump(pipe, box: "queue.Queue") -> None:
    try:
        for line in pipe:
            box.put(line)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _kill(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def cancel_running() -> str:
    proc = _current.get("proc")
    if proc is None or proc.poll() is not None:
        return "当前没有正在运行的转写任务。"
    _kill(proc)
    return "已发送终止信号，正在中断当前任务。"


def run_stream(req: RunRequest) -> Iterator[dict]:
    """生成器：先不断 yield 进度，最后 yield 一次带 result 的终局事件。"""
    backend, note = resolve_backend(req.backend)
    ok, why = backend_status(backend)
    if not ok:
        raise EngineError(why)

    binary = BACKEND_DIRS[backend] / "moss-transcribe"
    model = Path(req.model).expanduser()
    if not model.is_file():
        raise EngineError(f"模型文件不存在：{model}")
    if model.suffix.lower() != ".gguf":
        raise EngineError(f"模型必须是 .gguf 文件，当前是：{model}")
    if not Path(req.wav).is_file():
        raise EngineError(f"音频文件不存在：{req.wav}")
    if req.output_format not in OUTPUT_FORMATS:
        raise EngineError(f"不支持的输出格式：{req.output_format}")

    req.model, req.wav = str(model), str(req.wav)
    cmd = build_cmd(req, binary)
    env = build_env(req, backend)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=env, cwd=str(ROOT), start_new_session=True,
    )
    _current["proc"] = proc

    out_box: "queue.Queue" = queue.Queue()
    err_box: "queue.Queue" = queue.Queue()
    log_lines: list[str] = []
    out_lines: list[str] = []

    t_out = threading.Thread(target=_pump, args=(proc.stdout, out_box), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, err_box), daemon=True)
    t_out.start()
    t_err.start()

    t0 = time.time()

    def drain() -> None:
        while True:
            try:
                log_lines.append(err_box.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                out_lines.append(out_box.get_nowait())
            except queue.Empty:
                break

    try:
        while True:
            drain()
            if proc.poll() is not None:
                break
            if req.timeout and req.timeout > 0 and (time.time() - t0) > req.timeout:
                _kill(proc)
                drain()
                t_out.join(2)
                t_err.join(2)
                yield {
                    "done": True,
                    "result": RunResult(
                        ok=False, log="".join(log_lines), error="任务超时，已强制终止。",
                        elapsed=time.time() - t0, cmd=cmd, backend=backend,
                        binary=str(binary),
                    ),
                }
                return
            yield {
                "done": False, "elapsed": time.time() - t0,
                "log": "".join(log_lines), "backend": backend, "note": note,
            }
            time.sleep(0.3)

        # 进程已退出，把管道里剩下的读干净
        for _ in range(6):
            drain()
            if t_out.is_alive() or t_err.is_alive():
                time.sleep(0.15)
        drain()
        t_out.join(3)
        t_err.join(3)
        proc.wait()
    finally:
        # 用户点了中断 / 生成器被关闭时，连 ggml 起的孙进程一起杀掉
        _kill(proc)
        _current["proc"] = None

    elapsed = time.time() - t0
    stdout_text = "".join(out_lines)
    log = "".join(log_lines)
    code = proc.returncode
    device_m = _DEVICE_RE.search(log)
    device = device_m.group(1) if device_m else ""

    result = RunResult(
        ok=False, log=log, exit_code=code, elapsed=elapsed, cmd=cmd,
        backend=backend, binary=str(binary), device=device,
    )

    if code == 0:
        if not stdout_text.strip():
            result.error = "核心返回了成功状态，但输出为空。"
        else:
            result.ok = True
            result.text = stdout_text
    elif code == 1:
        result.error = "转写失败（核心退出码 1：模型加载或推理出错）。"
    elif code == 2:
        result.error = "参数错误（核心退出码 2：命令行用法不对）。"
    elif code is not None and code < 0:
        sig = -code
        hint = ""
        if backend == "rocm":
            hint = ("ROCm 后端崩溃，常见原因是 HSA_OVERRIDE_GFX_VERSION 与你的 GPU 不匹配，"
                    "或者 /dev/kfd 权限不足（需要 render/video 组）。可以试试把后端切成 Vulkan。")
        elif backend == "vulkan":
            hint = "Vulkan 后端崩溃，常见原因是驱动或 shader 缓存目录不可写，可以试试把后端切成 ROCm。"
        result.error = f"核心被信号 {sig} 终止（{'段错误 SIGSEGV' if sig == 11 else '异常退出'}）。" + hint
    else:
        result.error = f"核心异常退出，退出码 {code}。"

    yield {"done": True, "result": result}


def dump_text(req: RunRequest, text: str, audio_path: str) -> Path:
    """把结果落盘到 runtime/out/，供界面重复下载。"""
    out_dir = ROOT / "runtime" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem[:60] or "transcript"
    path = out_dir / (stem + FORMAT_SUFFIX.get(req.output_format, ".txt"))
    path.write_text(text, encoding="utf-8")
    return path


def dump_segmented(text: str, audio_path: str) -> Path:
    """分段合并后的结果落盘到 runtime/out/，文件名带 _seg 后缀，始终是 JSON。"""
    out_dir = ROOT / "runtime" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem[:60] or "transcript"
    path = out_dir / (stem + "_seg.json")
    path.write_text(text, encoding="utf-8")
    return path


def parse_speaker_text(text: str) -> list[dict]:
    """把引擎的 JSON（分段列表）收敛成 [{speaker, text}]，丢弃时间戳等无关字段。

    既用于单段结果，也用于分段模式下逐段提取后合并——满足「每条结果仅需
    speaker 与 text 两个字段」且「关闭时间标注」的要求。
    """
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for seg in data:
        if not isinstance(seg, dict):
            continue
        out.append({"speaker": seg.get("speaker", ""), "text": seg.get("text", "")})
    return out
