# mossweb — moss-transcribe.cpp 的 Gradio Web 界面

给 [moss-transcribe.cpp](https://github.com/localai-org/moss-transcribe.cpp) 套一层本地
Web 界面。模型单次前向就能同时产出**转写文本 + 说话人分离 + 时间戳**，这个界面负责把它
变得点几下就能用：上传任意格式音频、选后端与输出格式、拿到可下载的字幕。

* 界面只做转写与后端切换，没有做量化、模型信息页之类的附属功能。
* 由于 GitHub 对大小的限制， 所以说 ROCM 的核心后端的库无法上传;可以去我云盘里下载,或者参考上游自行编译!链接：https://1813527308.share.123pan.cn/123pan/PawSVv-zN9hd?pwd=8M1f# 提取码：8M1f
* 或者自行解压core.tar.gz替换,core里面的内容
* 本项目只是 web 界面有关核心的更多内容， 建议查看上游项目[moss-transcribe.cpp](https://github.com/localai-org/moss-transcribe.cpp)

---

## 快速开始

环境用 [uv](https://docs.astral.sh/uv/) 管理（仓库里已带 `pyproject.toml`）：

```bash
cd /home/ylhcq/Application/mossweb

# 建虚拟环境并装依赖（只需一次）
uv venv --python 3.12
uv pip install gradio

# 启动
uv run app.py
```

默认跑在 <http://127.0.0.1:7860>。可选参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | 监听地址，改成 `0.0.0.0` 可让局域网访问 |
| `--port` | `7860` | 端口 |
| `--share` | 关 | 生成 Gradio 公网临时链接 |
| `--open` | 关 | 启动后自动打开浏览器 |

也可以直接用虚拟环境里的解释器跑：`.venv/bin/python app.py`。

图省事的话用 `launcher.sh`，它会自动切到脚本所在目录、激活虚拟环境再启动，
从任何路径调用都可以，多余参数原样透传给 `app.py`：

```bash
./launcher.sh                  # 默认端口
./launcher.sh --port 8080      # 换端口
./launcher.sh --host 0.0.0.0   # 允许局域网访问
```

**注意**：Gradio 的主题是在 `app.py` 的 `launch()` 里配的，所以请用 `python app.py`
启动；如果用 `gradio app.py` 热重载模式，`main()` 不会执行，主题配置会被跳过。

---

## 目录结构

```
mossweb/
├── app.py              Gradio 界面：布局、输入校验、主流程编排
├── engine.py           命令行封装：拼命令、组装环境变量、流式日志、可取消
├── audio.py            音频预处理：ffprobe 探时长 + ffmpeg 转 WAV
├── config.py           设置持久化（mossweb.settings.json，原子写）
├── pyproject.toml      依赖声明（[tool.uv] package = false 表示这不是待构建的包）
├── launcher.sh         一键启动：切目录 + 激活 venv + 跑 app.py
├── core/
│   ├── rocm/           ROCm/HIP 编译的核心（moss-transcribe + libmoss-transcribe.so）
│   └── vulkan/         Vulkan 编译的核心
├── model/              放 .gguf 模型，界面只枚举这个目录
└── runtime/            运行时产物
    ├── tmp/            转码中间产物（每次启动自动清空）
    └── out/            转写结果文件，保留供重复下载
```

---

## 界面能调什么

### 输入

- **音频**：任意 ffmpeg 能解的格式（mp3 / m4a / flac / ogg / mp4 / wav …）。
  引擎只认 WAV，所以非 WAV 会先自动转码成 16 kHz 单声道 `pcm_s16le`。
- **模型**：下拉只枚举 `model/` 目录。也可以直接粘贴任意绝对路径（`allow_custom_value`）。
  项目内的模型显示成 `./mossweb/model/xxx.gguf · 618.1 MB` 这种相对形式，
  但传给子进程的值始终是绝对路径。
  模型的获取方式自行查看常见问题。

### 参数

| 控件 | 对应 | 说明 |
| --- | --- | --- |
| 输出格式 | `--format` | `text` 原始流 / `srt` / `ass` / `json`，默认 `srt` |
| `--max-new` | `--max-new N` | **生成 token 的上限**，不是目标长度。`≤0` 表示用 GGUF 里自带的默认值（推荐） |
| 后端 | `core/<后端>/` | `auto` / `rocm` / `vulkan` |
| `HSA_OVERRIDE_GFX_VERSION` | 环境变量 | 仅 ROCm，默认 `1150`，进程启动前自动导出，可改且会记住 |
| `MTD_DEVICE` | 环境变量 | `auto` / `cpu` / `gpu` |
| `MTD_THREADS` | 环境变量 | CPU 线程数，默认 8（解码是内存带宽瓶颈，线程不是越多越好） |
| 音频时长上限 | — | 默认 60 分钟，超出直接拒绝并提示 |
| 任务超时 | — | 默认 120 分钟，超时强杀进程组 |

关于 `--max-new`：模型是自回归地把文本、时间戳、说话人标签当成一个 token 流往外吐，
这个参数就是给这个流设天花板，主要用来防跑飞和控成本。默认值来自 GGUF 元数据
`mtd.default_max_new_tokens`（取不到就回退 5120）。**平时不用动**，只有两种场景要改：
长音频被截断就调大，模型卡住一直生成就调小当保险丝。

### 输出

- 转写结果文本框（`text` / `srt` / `ass` / `json` 原文）
- 分段表格（仅 `json` 格式会填充，展示 `# / 开始 / 结束 / 说话人 / 文本`）
- 下载按钮（结果同时落在 `runtime/out/` 下）
- 核心 stderr 日志（能看出实际跑在 CPU 还是 GPU 上）

### 分段处理（长音频）

勾选「分段处理」后，界面会先用 ffmpeg 把音频按「切片时长」（默认 60 秒，即每分钟一段）
切成分段，再**逐段调用核心转写**、最后把各段的字幕拼接合并。这样长音频不必一次性塞进
显存/内存，适合设备性能吃紧或音频很长的情况。

开启分段处理时有两个固定行为：

- **强制 JSON 输出**：结果以 JSON 返回，每条只包含 `speaker`（说话人）和 `text`（文本）
  两个字段，方便程序直接消费。
- **关闭时间标注**：各切片的时间戳在切分边界处并不准确，因此合并结果里**不含开始/结束时间**，
  分段表格也只显示 `# / 说话人 / 文本` 三列。

切片时长可调（最小 10 秒）。段越多、模型重复加载次数越多，请按需权衡。

> **已知限制**：说话人标签是**每段独立**做分离的，切片边界处的说话人编号会在不同段之间
> 重置，无法跨段对齐（例如第 2 段的 `S01` 不一定等于第 1 段的 `S01`）。这是逐段处理的
> 固有代价；介意的话用非分段模式，可拿到带时间戳、全局一致的分离结果。

---

## 后端是怎么回事

**后端是编译期绑定的**：`core/rocm/` 里的二进制只能跑 HIP/ROCm，`core/vulkan/` 里的
只能跑 Vulkan。所以界面上"切换后端"实际是切换用哪个目录下的二进制，不是同一个程序运行时
切。

两个 CLI 的 ELF **RUNPATH 指向原仓库的 `build-hip/`、`build-vk/`**，所以引擎启动时强制设了
`LD_LIBRARY_PATH=core/<后端>`（RUNPATH 优先级低于 `LD_LIBRARY_PATH`，这么设能覆盖掉），
否则会去加载原仓库里的 `.so`，脱离工作目录就跑不了。

`auto` 的选择逻辑：检测到 `/dev/kfd` 且 `/dev/dri` 存在就选 ROCm，否则选 Vulkan。

### ROCm 的 GFX 版本覆盖

选 ROCm（或 `auto` 选中 ROCm）时，会在 `Popen(env=...)` 里注入
`HSA_OVERRIDE_GFX_VERSION`（默认 `1150`）。必须在 **execve 之前**注入，也就是在
ROC 运行时 `dlopen` 之前——用 subprocess 的 `env=` 天然满足这一点。
值可以在界面上改，会写进 `mossweb.settings.json`，下次启动沿用。

界面「环境与自检」那一栏会显示两个后端是否可用、以及有没有检测到 `/dev/kfd` 和 `/dev/dri`。

---

## 转写要跑多久

CPU 8 线程实测（F32 模型，数据来自上游 README）：11 秒音频约 6.5 秒，132 秒音频约 103 秒。
**RTF 会随音频变长而上升**——解码是自回归的，上下文随音频增长。所以长音频要有心理准备，
模型本身的目标运行环境是 GPU。

模型 mmap 加载约 1 秒，每次转写都会重新加载（用子进程调用的代价，换来的是 GPU 崩溃不会
拖垮 Web 服务）。界面状态栏会显示已耗时和最终 RTF。

---

## 输入校验与错误提示

界面会拦下这些情况并给出可读提示：

- 没上传音频 / 模型路径不存在 / 模型不是 `.gguf` / 输出格式非法
- 文件损坏或没有音轨（ffprobe 解析失败）
- 音频时长超过设定上限
- ffmpeg / ffprobe 不存在
- 后端二进制缺失或不可执行、动态库缺失
- 核心退出码语义：
  - `0` 成功
  - `1` 模型加载或推理失败
  - `2` 命令行用法错误
  - 负数 被信号杀死（如 `-11` 段错误，会额外提示常见原因）

转写中途可以点「取消」，引擎用 `start_new_session=True` + `os.killpg` 把 ggml 起的孙进程
一起杀掉。

---

## 常见问题

**选 ROCm 直接崩（退出码 -11）**
大概率是 `HSA_OVERRIDE_GFX_VERSION` 和你的 GPU 不匹配，或者 `/dev/kfd` 权限不够（需要在
`render` / `video` 组里）。先试试改 GFX 版本，或者直接切成 Vulkan。

**两个后端都不可用**
检查 `core/rocm/` 和 `core/vulkan/` 下是否都有 `moss-transcribe` 和 `libmoss-transcribe.so`，
以及 `moss-transcribe` 有没有执行权限（`chmod +x`）。

**提示找不到模型**
`model/` 下没有 `.gguf`。把模型放进去点「刷新」，或者直接在下拉里填绝对路径。
获取GGUF模型的方法 GGUF 在 [mudler/moss-transcribe.cpp-gguf](https://huggingface.co/mudler/moss-transcribe.cpp-gguf)。
123云盘镜像(只有f32和q5_k)：链接：https://1813527308.share.123pan.cn/123pan/PawSVv-Lgzhd?pwd=A2V1# 提取码：A2V1
有关模型的更多内容包括获取转换， 请参考上游项目[moss-transcribe.cpp](https://github.com/localai-org/moss-transcribe.cpp)

**上传 mp3 报转码失败**
需要系统里有 `ffmpeg` / `ffprobe`（当前环境是 8.1.2）。引擎只认 WAV，没有 ffmpeg 就只能
手动传 WAV。

---

## 已知限制

- 引擎只认 WAV，其余格式依赖 ffmpeg 预转码。
- 不支持 VTT 输出（核心只支持 `text` / `srt` / `ass` / `json`）。
- 没有语言、温度、翻译之类的开关——这些要么写死在 GGUF 元数据里，要么模型本身不支持
  （解码是贪心 argmax，没有采样）。
- 并发限制为 1（`demo.queue(default_concurrency_limit=1)`），避免多个任务抢 GPU。
- `runtime/tmp/` 每次启动清空，`runtime/out/` 会一直累积，需要自己清理。
