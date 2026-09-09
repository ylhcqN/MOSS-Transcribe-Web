"""moss-transcribe.cpp 的 Gradio Web 界面。

运行：
    uv run app.py                    # 默认 http://127.0.0.1:7860
    uv run app.py --port 8080 --host 0.0.0.0 --share
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import gradio as gr

import audio as audio_mod
import config as config_mod
import engine

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "moss-transcribe-q5_k.gguf"
# 只枚举项目自己的 model/ 目录，不碰别处的临时工作副本
MODEL_DIRS = [ROOT / "model"]

TABLE_HEADERS = ["#", "开始", "结束", "说话人", "文本"]


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def _size_str(p: Path) -> str:
    try:
        mb = p.stat().st_size / 1024 / 1024
    except OSError:
        return "?"
    return f"{mb:.1f} MB" if mb < 1024 else f"{mb / 1024:.2f} GB"


def proj_display(p: Path) -> str:
    """项目内的路径显示成 ./mossweb/... 这种相对形式，项目外的照旧显示绝对路径。"""
    rp = p.resolve()
    try:
        return "./" + rp.relative_to(ROOT.parent).as_posix()
    except ValueError:
        return str(rp)


def _display(p: Path) -> str:
    return f"{proj_display(p)} · {_size_str(p)}"


def discover_models() -> list[tuple[str, str]]:
    """返回 (显示名, 绝对路径) 列表。值用绝对路径，避免子进程 cwd 变化导致找不到模型。"""
    seen, out = set(), []
    for d in MODEL_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.gguf")):
            ap = str(p.resolve())
            if ap in seen:
                continue
            seen.add(ap)
            out.append((_display(p.resolve()), ap))
    return out


def pick_model(*candidates: str) -> str:
    """从候选里挑一个真实存在的模型路径。"""
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    models = discover_models()
    return models[0][1] if models else ""


def fmt_time(sec: float) -> str:
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return "-"
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def segments_to_rows(text: str) -> list[list]:
    import json
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    rows = []
    for i, seg in enumerate(data, 1):
        if not isinstance(seg, dict):
            continue
        rows.append([
            i,
            fmt_time(seg.get("start", 0)),
            fmt_time(seg.get("end", 0)),
            seg.get("speaker", ""),
            seg.get("text", ""),
        ])
    return rows


def join_cmd(cmd: list[str]) -> str:
    return " ".join(f"'{c}'" if " " in c else c for c in cmd)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def do_transcribe(audio_path, model, backend, hsa, out_fmt, max_new,
                  device, threads, max_minutes, timeout_minutes):
    status = gr.update
    # --- 输入校验 ---
    if not audio_path:
        gr.Warning("请先上传一个音频文件。")
        yield ("❌ 请先上传音频文件。", "", "", [], None, "")
        return
    if not model or not Path(str(model).strip()).is_file():
        gr.Warning("模型路径无效。")
        yield ("❌ 模型文件不存在，请检查路径。", "", "", [], None, "")
        return
    if out_fmt not in engine.OUTPUT_FORMATS:
        gr.Warning("输出格式不合法。")
        yield ("❌ 输出格式不合法。", "", "", [], None, "")
        return
    try:
        max_new = int(max_new)
    except (TypeError, ValueError):
        max_new = -1
    if max_new < -1:
        max_new = -1

    # 持久化本次设置（HSA 值下次启动沿用）
    try:
        config_mod.save({
            "backend": backend, "hsa_override_gfx_version": hsa,
            "mtd_device": device, "mtd_threads": int(threads or 0),
            "output_format": out_fmt, "max_new": max_new,
            "max_audio_minutes": float(max_minutes or 0),
            "timeout_minutes": float(timeout_minutes or 0),
            "last_model": str(model),
        })
    except Exception:
        pass

    # --- 音频预处理（引擎只认 WAV） ---
    try:
        wav, duration = audio_mod.ensure_wav(audio_path, float(max_minutes or 0))
    except audio_mod.AudioError as e:
        yield (f"❌ {e}", "", "", [], None, "")
        return
    except Exception as e:
        yield (f"❌ 音频预处理出错：{e}", "", "", [], None, "")
        return

    req = engine.RunRequest(
        model=str(model).strip(), wav=str(wav), backend=backend,
        output_format=out_fmt, max_new=max_new,
        hsa_override_gfx_version=str(hsa or engine.DEFAULT_HSA_OVERRIDE_GFX_VERSION),
        mtd_device=device, mtd_threads=int(threads or 0),
        timeout=float(timeout_minutes or 0) * 60.0,
    )

    try:
        backend_used = engine.resolve_backend(backend)[0]
    except engine.EngineError as e:
        yield (f"❌ {e}", "", "", [], None, "")
        return

    head = (f"音频 {duration:.1f}s（{duration / 60:.1f} 分钟）｜ 后端 {backend_used} ｜ "
            f"格式 {out_fmt}")
    if backend_used == "rocm":
        head += f" ｜ HSA_OVERRIDE_GFX_VERSION={req.hsa_override_gfx_version}"

    try:
        for ev in engine.run_stream(req):
            if not ev.get("done"):
                yield (f"⏳ 转写中… 已用 {ev['elapsed']:.1f}s ｜ {head}",
                       ev.get("log", ""), "", [], None, "")
                continue

            r: engine.RunResult = ev["result"]
            log = r.log or ""
            info_md = (
                f"**命令**：`{join_cmd(r.cmd)}`\n\n"
                f"**二进制**：`{r.binary}`\n\n"
                f"**LD_LIBRARY_PATH**：`{engine.BACKEND_DIRS[r.backend]}`\n\n"
                f"**退出码**：`{r.exit_code}`"
            )

            if not r.ok:
                err = f"❌ {r.error}"
                if r.log.strip():
                    err += f"\n\n核心日志：\n```\n{r.log.strip()[-1500:]}\n```"
                yield (err, log, "", [], None, info_md)
                return

            out_file = engine.dump_text(req, r.text, str(wav))
            rows = segments_to_rows(r.text) if out_fmt == "json" else []

            dev = f" ｜ 实际设备 {r.device}" if r.device else ""
            ok_md = (f"✅ 完成 ｜ 后端 {r.backend}{dev} ｜ 耗时 {r.elapsed:.1f}s ｜ "
                     f"音频 {duration:.1f}s（RTF {r.elapsed / max(duration, 0.01):.2f}）")
            if out_fmt == "json" and not rows:
                ok_md += "\n\n⚠️ JSON 解析为空，可能模型没有输出有效分段，请看「转写结果」原始内容。"
            yield (ok_md, log, r.text, rows, str(out_file), info_md)
            return
    except engine.EngineError as e:
        yield (f"❌ {e}", "", "", [], None, "")
        return
    except Exception as e:
        yield (f"❌ 未预期的错误：{type(e).__name__}: {e}", "", "", [], None, "")
        return


def do_cancel():
    return engine.cancel_running()


def toggle_hsa(backend: str):
    return gr.update(visible=backend in ("rocm", "auto"))


def refresh_models():
    models = discover_models()
    values = [v for _, v in models]
    cur = pick_model(str(DEFAULT_MODEL), *(values or []))
    return gr.update(choices=models, value=cur)


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

def build_ui(settings: dict) -> gr.Blocks:
    with gr.Blocks(title="MOSS Transcribe Web", analytics_enabled=False) as demo:
        gr.Markdown(
            "# MOSS Transcribe Web\n"
            "基于 [moss-transcribe.cpp](https://github.com/localai-org/moss-transcribe.cpp) 的本地转写界面："
            "单次前向同时产出 **转写文本 + 说话人分离 + 时间戳**。\n\n"
            "上传任意格式的音频 → 自动用 ffmpeg 转成引擎唯一认的 WAV → 调用 `core/<后端>/moss-transcribe` → "
            "输出 text / srt / ass / json。"
        )

        with gr.Row():
            # ---------------- 左：输入 ----------------
            with gr.Column(scale=4):
                audio_in = gr.Audio(
                    sources=["upload"], type="filepath",
                    label="音频文件（mp3 / m4a / flac / ogg / wav / mp4 … 会自动转码）",
                )

                init_models = discover_models()
                init_values = [v for _, v in init_models]
                init_model = pick_model(
                    str(settings.get("last_model") or ""),
                    str(DEFAULT_MODEL), *(init_values or []),
                )
                if not init_models:
                    gr.Markdown(
                        f"⚠️ 在 `{proj_display(MODEL_DIRS[0])}` 下没有找到任何 .gguf 文件，"
                        "把模型放进去后点「刷新」，或者在下面直接填绝对路径。"
                    )

                with gr.Row():
                    model_dd = gr.Dropdown(
                        choices=init_models, value=init_model,
                        allow_custom_value=True, filterable=True,
                        label="模型 (.gguf)", scale=5,
                        info=f"只扫描项目目录：{proj_display(MODEL_DIRS[0])}（也可直接粘贴其它绝对路径）",
                    )
                    refresh_btn = gr.Button("刷新", scale=1, min_width=70)

                with gr.Row():
                    out_fmt = gr.Radio(
                        choices=list(engine.OUTPUT_FORMATS),
                        value=settings.get("output_format", "srt"),
                        label="输出格式",
                    )
                    max_new_nb = gr.Number(
                        value=settings.get("max_new", -1), precision=0,
                        label="--max-new：生成 token 上限",
                        info="上限，不是目标长度。≤0 表示用 GGUF 里自带的默认值（推荐，通常够长）。"
                             "调得太小会把转写结果硬截断；官方示例给的是 4096。",
                    )

                gr.Markdown("### 核心 / 后端")
                backend_radio = gr.Radio(
                    choices=["auto", "rocm", "vulkan"],
                    value=settings.get("backend", "auto"),
                    label="后端（核心是编译期绑定后端的，切换即切换二进制目录）",
                )
                hsa_tb = gr.Textbox(
                    value=settings.get("hsa_override_gfx_version",
                                       engine.DEFAULT_HSA_OVERRIDE_GFX_VERSION),
                    label="HSA_OVERRIDE_GFX_VERSION（仅 ROCm，进程启动前自动导出）",
                    visible=settings.get("backend", "auto") in ("rocm", "auto"),
                    info="ROCm 需要在 dlopen 之前声明目标 GFX 版本；这个值会记住，下次启动沿用。",
                )
                with gr.Row():
                    device_dd = gr.Dropdown(
                        choices=["auto", "cpu", "gpu"],
                        value=settings.get("mtd_device", "auto"),
                        label="MTD_DEVICE",
                    )
                    threads_nb = gr.Number(
                        value=settings.get("mtd_threads", 8), precision=0,
                        minimum=1, label="MTD_THREADS",
                    )
                with gr.Row():
                    max_min_nb = gr.Number(
                        value=settings.get("max_audio_minutes", 60),
                        label="音频时长上限（分钟）",
                    )
                    timeout_nb = gr.Number(
                        value=settings.get("timeout_minutes", 120),
                        label="任务超时（分钟）",
                    )

                with gr.Row():
                    run_btn = gr.Button("开始转写", variant="primary", scale=3)
                    cancel_btn = gr.Button("取消", scale=1)

                status_md = gr.Markdown("等待任务…")

                with gr.Accordion("环境与自检", open=False):
                    env_md = gr.Markdown(engine.backend_summary())
                    gr.Markdown(
                        f"- ffmpeg：`{shutil.which('ffmpeg') or '未找到'}`\n"
                        f"- ffprobe：`{shutil.which('ffprobe') or '未找到'}`\n"
                        f"- 配置文件：`{config_mod.CONFIG_PATH}`"
                    )

            # ---------------- 右：输出 ----------------
            with gr.Column(scale=6):
                out_text = gr.Textbox(
                    label="转写结果", lines=18, max_lines=40, buttons=["copy"],
                )
                out_table = gr.Dataframe(
                    headers=TABLE_HEADERS, label="分段表（仅 json 格式会填充）",
                    interactive=False, wrap=True,
                )
                dl_file = gr.File(label="下载结果文件")
                log_tb = gr.Textbox(label="核心日志 (stderr)", lines=6, max_lines=6)
                with gr.Accordion("本次执行的命令与环境", open=False):
                    runinfo_md = gr.Markdown("")

        # ---------------- 事件 ----------------
        refresh_btn.click(refresh_models, outputs=[model_dd])
        backend_radio.change(toggle_hsa, inputs=[backend_radio], outputs=[hsa_tb])

        run_btn.click(
            do_transcribe,
            inputs=[audio_in, model_dd, backend_radio, hsa_tb, out_fmt, max_new_nb,
                    device_dd, threads_nb, max_min_nb, timeout_nb],
            outputs=[status_md, log_tb, out_text, out_table, dl_file, runinfo_md],
        )
        cancel_btn.click(do_cancel, outputs=[status_md])

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="moss-transcribe.cpp Gradio 界面")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网访问链接")
    ap.add_argument("--open", action="store_true", help="自动打开浏览器")
    args = ap.parse_args()

    settings = config_mod.load()
    audio_mod.cleanup()  # 清掉上次遗留的转码中间产物，runtime/out 里的结果保留
    missing = [n for n in ("rocm", "vulkan") if not engine.backend_status(n)[0]]
    if len(missing) == 2:
        print("警告：core/rocm 与 core/vulkan 下都没有可用的核心，界面能起来但转写会失败。",
              file=sys.stderr)

    demo = build_ui(settings)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host, server_port=args.port,
        share=args.share, inbrowser=args.open,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.gray,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
