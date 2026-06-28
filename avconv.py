#!/usr/bin/env python3
"""
avconv – Convierte vídeos a H.264 usando GPU NVIDIA (NVENC).

Por defecto busca AV1; opcionalmente cualquier codec distinto de H.264.

Uso:
  avconv                          # directorio actual, interactivo
  avconv -d ~/Videos              # directorio específico
  avconv --all-codecs             # HEVC, VP9, AV1, etc.
  avconv --av1-only               # solo AV1 (equivalente al defecto)
  avconv -j 4                     # 4 conversiones en paralelo
  avconv -y                       # saltar confirmación
  avconv --quality 23             # priorizar calidad (sin preguntar)
  avconv --size-match             # tamaño similar (sin preguntar)
  avconv --edit-friendly          # optimizar para edición (sin preguntar)
  avconv --dry-run                # solo listar vídeos, no convertir
  avconv -f                       # sobrescribir archivos _h264 existentes
"""

import subprocess
import sys
import os
import json
import re
import unicodedata
import time
import threading
import signal
import argparse
import shutil
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from datetime import timedelta


@dataclass
class ConvertOptions:
    """Opciones de codificación elegidas antes de convertir."""
    quality_cq: Optional[int]   # None = tamaño similar al original
    edit_friendly: bool         # keyframes frecuentes + faststart
    replace_originals: bool     # True = sustituir AV1 in situ; False = crear *_h264.*

# ─── Terminal styling ───────────────────────────────────────────────────────

class _Term:
    """ANSI colors + iconos. Sin dependencias externas."""

    def __init__(self) -> None:
        self.enabled = False

    def configure(self, *, no_color: bool = False) -> None:
        if no_color or os.environ.get('NO_COLOR'):
            self.enabled = False
            return
        if not sys.stdout.isatty():
            self.enabled = False
            return
        if sys.platform == 'win32':
            self._enable_windows_vt()
        self.enabled = True

    @staticmethod
    def _enable_windows_vt() -> None:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (OSError, AttributeError):
            pass

    def _c(self, text: str, *codes: str) -> str:
        if not self.enabled:
            return text
        return ''.join(codes) + text + '\033[0m'

    def bold(self, t: str) -> str:
        return self._c(t, '\033[1m')

    def dim(self, t: str) -> str:
        return self._c(t, '\033[2m')

    def italic(self, t: str) -> str:
        return self._c(t, '\033[3m')

    def underline(self, t: str) -> str:
        return self._c(t, '\033[4m')

    def red(self, t: str) -> str:
        return self._c(t, '\033[91m')

    def green(self, t: str) -> str:
        return self._c(t, '\033[92m')

    def yellow(self, t: str) -> str:
        return self._c(t, '\033[93m')

    def blue(self, t: str) -> str:
        return self._c(t, '\033[94m')

    def magenta(self, t: str) -> str:
        return self._c(t, '\033[95m')

    def cyan(self, t: str) -> str:
        return self._c(t, '\033[96m')

    def white(self, t: str) -> str:
        return self._c(t, '\033[97m')

    def bg_blue(self, t: str) -> str:
        return self._c(t, '\033[44m', '\033[97m', '\033[1m')

    def gradient_line(self, text: str) -> str:
        if not self.enabled:
            return text
        palette = ['\033[96m', '\033[94m', '\033[95m', '\033[93m']
        out: list[str] = []
        for i, ch in enumerate(text):
            out.append(palette[i % len(palette)] + ch)
        out.append('\033[0m')
        return ''.join(out)


term = _Term()

# Iconos
ICO = {
    'app': '🎬',
    'gpu': '⚡',
    'dir': '📁',
    'scan': '🔍',
    'ok': '✔',
    'fail': '✘',
    'warn': '⚠',
    'run': '▶',
    'wait': '⏳',
    'stats': '📊',
    'quality': '🎯',
    'parallel': '🔀',
    'out': '💾',
    'film': '🎞',
    'time': '⏱',
    'size': '📦',
}


def _clear_screen() -> None:
    if term.enabled:
        print('\033[2J\033[H', end='', flush=True)
    else:
        os.system('cls' if os.name == 'nt' else 'clear')


def _box_top(width: int = 58) -> str:
    return f"  {term.cyan('╭' + '─' * width + '╮')}"


def _box_bot(width: int = 58) -> str:
    return f"  {term.cyan('╰' + '─' * width + '╯')}"


def _strip_ansi(s: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', s)


def _char_display_width(ch: str) -> int:
    if not ch or unicodedata.combining(ch):
        return 0
    o = ord(ch)
    if o < 32 or o == 0x7F:
        return 0
    if unicodedata.east_asian_width(ch) in ('W', 'F'):
        return 2
    # Emoji / símbolos que ocupan 2 columnas en consola Windows
    if 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x2300 <= o <= 0x23FF:
        return 2
    return 1


def _display_width(s: str) -> int:
    return sum(_char_display_width(ch) for ch in _strip_ansi(s))


def _box_line(content: str, width: int = 58) -> str:
    pad = max(0, width - _display_width(content))
    return f"  {term.cyan('│')} {content}{' ' * pad} {term.cyan('│')}"


def _pad_col(text: str, width: int, style: Callable[[str], str] | None = None) -> str:
    return _cell(text, width, 'left', style)


def _cell(
    text: str,
    width: int,
    align: str = 'left',
    style: Callable[[str], str] | None = None,
) -> str:
    """Celda con ancho visual fijo (emojis/ANSI no desalinean columnas)."""
    dw = _display_width(text)
    pad = max(0, width - dw)
    core = style(text) if style and term.enabled else text
    if align == 'right':
        return ' ' * pad + core
    return core + ' ' * pad


def _truncate_display(text: str, max_width: int) -> str:
    if _display_width(text) <= max_width:
        return text
    ell = '…'
    budget = max_width - _display_width(ell)
    out: list[str] = []
    w = 0
    for ch in text:
        cw = _char_display_width(ch)
        if w + cw > budget:
            break
        out.append(ch)
        w += cw
    return ''.join(out) + ell


# Anchos de columnas de la tabla de vídeos
_TCOL_NUM, _TCOL_NAME, _TCOL_RES, _TCOL_DUR, _TCOL_SZ, _TCOL_CODEC = (
    3, 40, 12, 9, 8, 5,
)


def _table_row(cells: list[str]) -> str:
    return ''.join(cells)


def _kv(icon: str, label: str, value: str) -> str:
    return (
        f"  {icon}  {term.dim(label + ':')}  "
        f"{term.bold(term.white(value))}"
    )


def _progress_bar(pct: float, length: int = 24) -> str:
    filled = int(pct / 100 * length)
    empty = length - filled
    if term.enabled:
        if pct >= 100:
            return term.green('█' * length)
        if pct > 0:
            return term.cyan('█' * filled) + term.dim('░' * empty)
        return term.dim('░' * length)
    return '█' * filled + '░' * empty


def _print_error(msg: str) -> None:
    print(f"\n  {term.red(ICO['fail'] + ' ERROR')}  {term.bold(term.red(msg))}\n")


def _print_warn(msg: str) -> None:
    print(f"  {term.yellow(ICO['warn'])}  {term.yellow(msg)}")


def _print_info(msg: str) -> None:
    print(f"  {term.blue('ℹ')}  {msg}")


def _print_success(msg: str) -> None:
    print(f"  {term.green(ICO['ok'])}  {term.green(msg)}")


def _prompt(question: str) -> str:
    print(f"  {term.cyan('?')}  {term.bold(question)} ", end='', flush=True)
    return input().strip().lower()


def _prompt_yes_no(question: str, *, default: bool = False) -> bool:
    hint = '[S/n]' if default else '[s/N]'
    try:
        resp = _prompt(f"{question} {hint}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {term.dim('Abortado.')}")
        sys.exit(0)
    if not resp:
        return default
    return resp in ('s', 'si', 'y', 'yes')


def _prompt_choice(question: str, options: list[tuple[str, str]], default: int = 0) -> int:
    """Menú numerado. Devuelve índice elegido."""
    print()
    print(f"  {term.cyan('?')}  {term.bold(question)}")
    for i, (label, desc) in enumerate(options):
        mark = term.green('●') if i == default else term.dim('○')
        num = term.bold(str(i + 1))
        print(f"     {mark}  {num}  {term.white(label)}  {term.dim(desc)}")
    print()
    try:
        resp = _prompt(f"Elige [1-{len(options)}] (defecto {default + 1}):")
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {term.dim('Abortado.')}")
        sys.exit(0)
    if not resp:
        return default
    if resp.isdigit():
        idx = int(resp) - 1
        if 0 <= idx < len(options):
            return idx
    _print_warn("Opción no válida, usando defecto.")
    return default

# ─── Config ─────────────────────────────────────────────────────────────────

DEFAULT_QUALITY = 23          # CQ en modo calidad (archivos más grandes)
DEFAULT_CQ_SIZE_MATCH = 34    # techo CQ en modo tamaño similar
SIZE_MATCH_MAXRATE_FACTOR = 1.15
SIZE_MATCH_BUFSIZE_FACTOR = 2.0
EDIT_KEYFRAME_SEC = 1.0       # 1 keyframe por segundo en modo edición
COMPACT_PROGRESS_AT = 12      # por encima: vista resumida (no listar todos)
COMPACT_RECENT_DONE = 3       # completados recientes visibles en modo compacto
COMPACT_MAX_ERRORS = 8        # errores visibles en modo compacto
DEFAULT_PARALLEL = 2          # RTX 5070Ti tiene 2 NVENC encoders físicos
VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.m2ts'}
OUTPUT_TAG = '_h264'          # se añade antes de la extensión
H264_CODEC_NAMES = frozenset({'h264', 'avc', 'avc1'})
MIN_VIDEO_BYTES = 1_000_000   # ignorar archivos < 1 MB (falsos positivos)
SKIP_DIRS = {'.cache', 'node_modules', '.git', '__pycache__', '.venv',
             'venv', 'env', '.svn', '.hg', '.tox', '.pytest_cache',
             '.next', 'build', 'dist', 'target', 'bin', 'obj'}

# Procesos ffmpeg activos (para cancelación limpia con Ctrl+C)
_active_procs: set[subprocess.Popen] = set()
_active_procs_lock = threading.Lock()
_cancel_requested = threading.Event()


# ─── Data ───────────────────────────────────────────────────────────────────

@dataclass
class VideoInfo:
    path: Path
    codec: str
    width: int
    height: int
    duration_sec: float
    size_bytes: int
    bitrate_kbps: float
    video_bitrate_kbps: float
    fps: float

    @property
    def size_str(self) -> str:
        return _fmt_size(self.size_bytes)

    @property
    def dur_str(self) -> str:
        return str(timedelta(seconds=int(self.duration_sec)))

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def sidecar_path(self) -> Path:
        """Archivo de salida junto al original (*_h264.*)."""
        return self.path.with_stem(self.path.stem + OUTPUT_TAG)

    @property
    def output_path(self) -> Path:
        return self.sidecar_path

    def encode_target(self, replace_originals: bool) -> Path:
        """Ruta donde ffmpeg escribe (temporal si se reemplaza el original)."""
        if replace_originals:
            # Debe terminar en .mp4/.mkv/etc.; ffmpeg rechaza extensiones no estándar
            return self.path.with_name(f"{self.path.stem}.avconv_tmp{self.path.suffix}")
        return self.sidecar_path

    def final_path(self, replace_originals: bool) -> Path:
        """Ruta final del vídeo H.264 tras una conversión correcta."""
        if replace_originals:
            return self.path
        return self.sidecar_path

    @property
    def target_video_bitrate_kbps(self) -> float:
        if self.video_bitrate_kbps > 0:
            return self.video_bitrate_kbps
        if self.duration_sec > 0 and self.bitrate_kbps > 0:
            return max(self.bitrate_kbps * 0.85, 500)
        return 4000

    @property
    def gop_size(self) -> int:
        fps = self.fps if self.fps > 1 else 30
        return max(int(round(fps * EDIT_KEYFRAME_SEC)), 1)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02}m{s:02}s"
    return f"{m:02}m{s:02}s"


def _fmt_bitrate(kbps: float) -> str:
    if kbps >= 1000:
        return f"{kbps / 1000:.1f} Mbps"
    return f"{int(kbps)} kbps"


def _is_h264_codec(codec: str) -> bool:
    return codec.lower() in H264_CODEC_NAMES


def _codec_display(codec: str) -> str:
    c = codec.lower()
    if c in ('av1', 'av01'):
        return 'AV1'
    if c in ('hevc', 'h265'):
        return 'HEVC'
    if c in ('vp9', 'vp09'):
        return 'VP9'
    if c in ('vp8',):
        return 'VP8'
    if c in ('mpeg4', 'msmpeg4v3'):
        return 'MPEG4'
    return codec.upper()[:6]


def _should_convert_codec(codec: str, *, av1_only: bool) -> bool:
    if not codec or _is_h264_codec(codec):
        return False
    if av1_only:
        return codec.lower() in ('av1', 'av01')
    return True


def _estimate_video_bitrate_kbps(
    data: dict, vs: dict, size_bytes: int, duration_sec: float,
) -> float:
    if vs.get('bit_rate'):
        return float(vs['bit_rate']) / 1000
    audio_kbps = sum(
        float(s.get('bit_rate', 0)) / 1000
        for s in data.get('streams', [])
        if s.get('codec_type') == 'audio' and s.get('bit_rate')
    )
    if duration_sec > 0:
        total_kbps = (size_bytes * 8 / 1000) / duration_sec
        if audio_kbps > 0:
            return max(total_kbps - audio_kbps, 500)
        return max(total_kbps * 0.85, 500)
    return 0.0


def _check_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _subprocess_kwargs() -> dict:
    kw: dict = {
        'encoding': 'utf-8',
        'errors': 'replace',
    }
    if sys.platform == 'win32' and hasattr(subprocess, 'CREATE_NO_WINDOW'):
        kw['creationflags'] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return kw


def _register_proc(proc: subprocess.Popen) -> None:
    with _active_procs_lock:
        _active_procs.add(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    with _active_procs_lock:
        _active_procs.discard(proc)


def terminate_all_procs() -> None:
    with _active_procs_lock:
        procs = list(_active_procs)
    for proc in procs:
        try:
            proc.terminate()
        except OSError:
            pass


def _handle_interrupt(_signum, _frame) -> None:
    _cancel_requested.set()
    terminate_all_procs()


# ─── ffprobe ────────────────────────────────────────────────────────────────

def get_video_info(path: Path, *, av1_only: bool) -> Optional[VideoInfo]:
    """Extrae info del stream de vídeo con ffprobe.

    Devuelve None si el codec no aplica o hay error.
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(path),
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            **_subprocess_kwargs(),
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
    except Exception:
        return None

    # Buscar el primer stream de vídeo
    vs = None
    for s in data.get('streams', []):
        if s.get('codec_type') == 'video':
            vs = s
            break
    if not vs:
        return None

    codec = vs.get('codec_name', '').lower()
    if not _should_convert_codec(codec, av1_only=av1_only):
        return None

    fmt = data.get('format', {})
    duration = float(fmt.get('duration', vs.get('duration', 0)))
    size = int(fmt.get('size', os.path.getsize(path)))
    bitrate = float(fmt.get('bit_rate', 0)) / 1000 if fmt.get('bit_rate') else 0
    video_bitrate = _estimate_video_bitrate_kbps(data, vs, size, duration)

    fps_str = vs.get('r_frame_rate', '0/1')
    fps = 0.0
    if '/' in fps_str:
        try:
            n, d = fps_str.split('/')
            fps = float(n) / float(d)
        except Exception:
            fps = 0.0

    return VideoInfo(
        path=path,
        codec=codec,
        width=int(vs.get('width', 0)),
        height=int(vs.get('height', 0)),
        duration_sec=duration,
        size_bytes=size,
        bitrate_kbps=bitrate,
        video_bitrate_kbps=video_bitrate,
        fps=fps,
    )


# ─── Scanner ────────────────────────────────────────────────────────────────

def scan_directory(directory: Path, *, av1_only: bool) -> list[VideoInfo]:
    """Escanea recursivamente vídeos convertibles (AV1 o cualquier no-H.264).

    Salta directorios comunes no-video (.cache, node_modules, etc.)
    y archivos demasiado pequeños para ser vídeos reales.
    """
    label = 'AV1' if av1_only else 'convertibles'
    videos: list[VideoInfo] = []
    scanned = 0
    t0 = time.monotonic()

    for f in sorted(directory.rglob('*')):
        # Saltar directorios no-video en el camino
        try:
            rel = f.relative_to(directory)
            if any(part.startswith('.') or part in SKIP_DIRS
                   for part in rel.parts[:-1]):
                continue
        except ValueError:
            pass

        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        if OUTPUT_TAG in f.stem or '.avconv_tmp' in f.stem:
            continue

        # mínimo tamaño razonable para un vídeo
        stat = f.stat()
        if stat.st_size < MIN_VIDEO_BYTES:
            continue

        scanned += 1
        info = get_video_info(f, av1_only=av1_only)
        if info:
            videos.append(info)

        # Feedback cada ~5s si el escaneo va lento
        elapsed = time.monotonic() - t0
        if elapsed > 3 and scanned % 50 == 0:
            msg = (
                f"  {ICO['scan']}  {term.cyan('Escaneando…')} "
                f"{term.dim(f'{scanned} archivos · {len(videos)} {label}')}"
            )
            print(msg, end='\r', flush=True)

    return videos


# ─── GPU / NVENC detection ─────────────────────────────────────────────────

def check_nvenc() -> bool:
    """Verifica que h264_nvenc está disponible en ffmpeg."""
    try:
        r = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10,
            **_subprocess_kwargs(),
        )
        out = (r.stdout or '') + (r.stderr or '')
        return 'h264_nvenc' in out
    except Exception:
        return False


def auto_parallel(videos: list) -> int:
    """Determina el número óptimo de conversiones en paralelo.

    RTX 5070Ti tiene 2 NVENC encoders físicos → 2 como base.
    Si hay pocos vídeos, se limita a ese número.
    """
    base = DEFAULT_PARALLEL
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
            **_subprocess_kwargs(),
        )
        gpu_name = r.stdout.strip().lower() if r.returncode == 0 else ''
        # Las RTX 50-series tienen 2 NVENC (como las 40-series)
        if '5070' in gpu_name or '5060' in gpu_name:
            base = 2
        elif '5080' in gpu_name or '5090' in gpu_name:
            base = 3  # 5090 tiene 3 NVENC
        elif '4090' in gpu_name or '4080' in gpu_name or '4070' in gpu_name:
            base = 2
    except Exception:
        pass
    return min(base, len(videos))


def get_gpu_name() -> str:
    """Devuelve el nombre de la GPU NVIDIA, o '' si no se puede detectar."""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
            **_subprocess_kwargs(),
        )
        return r.stdout.strip() if r.returncode == 0 else ''
    except Exception:
        return ''


# ─── ffmpeg conversion ──────────────────────────────────────────────────────

def _build_cmd(in_path: Path, out_path: Path, info: VideoInfo, opts: ConvertOptions) -> list[str]:
    """Construye ffmpeg NVENC (decode CPU, encode GPU)."""
    cmd = [
        'ffmpeg', '-hide_banner', '-nostdin', '-y',
        '-i', str(in_path),
        '-map', '0:v:0',
        '-map', '0:a?',
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'vbr',
        '-profile:v', 'high',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        '-progress', 'pipe:1',
        '-nostats',
    ]

    if opts.quality_cq is not None:
        cmd.extend(['-cq', str(opts.quality_cq), '-b:v', '0'])
    else:
        kbps = info.target_video_bitrate_kbps
        bps = max(int(kbps * 1000), 500_000)
        cmd.extend([
            '-b:v', str(bps),
            '-maxrate', str(int(bps * SIZE_MATCH_MAXRATE_FACTOR)),
            '-bufsize', str(int(bps * SIZE_MATCH_BUFSIZE_FACTOR)),
            '-cq', str(DEFAULT_CQ_SIZE_MATCH),
        ])

    if opts.edit_friendly:
        cmd.extend([
            '-g', str(info.gop_size),
            '-bf', '0',
            '-forced-idr', '1',
        ])
        if out_path.suffix.lower() in ('.mp4', '.mov', '.m4v'):
            cmd.extend(['-movflags', '+faststart'])

    cmd.append(str(out_path))
    return cmd


def _safe_progress(
    on_progress: Callable[[str, float, float], None],
    name: str,
    pct: float,
    speed: float,
) -> None:
    try:
        on_progress(name, pct, speed)
    except Exception:
        pass


def _run_ffmpeg(
    cmd: list[str],
    info: VideoInfo,
    on_progress: Callable[[str, float, float], None],
) -> tuple[int, str]:
    """Ejecuta ffmpeg drenando stdout (progreso) y stderr (evita deadlock)."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **_subprocess_kwargs(),
        )
    except Exception as e:
        return -1, f"No se pudo iniciar ffmpeg: {e}"

    _register_proc(proc)
    start = time.monotonic()
    end_us = int(info.duration_sec * 1_000_000)
    stderr_lines: list[str] = []

    def _stderr_reader() -> None:
        assert proc.stderr is not None
        try:
            for line in iter(proc.stderr.readline, ''):
                stderr_lines.append(line)
                if _cancel_requested.is_set():
                    break
        except (OSError, ValueError):
            pass

    def _parse_progress_us(raw: str) -> Optional[int]:
        raw = raw.strip()
        if not raw or raw.upper() == 'N/A':
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _stdout_reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if _cancel_requested.is_set():
                    break
                line = line.strip()
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k != 'out_time_us':
                    continue
                us = _parse_progress_us(v)
                if us is None:
                    continue
                pct = min(100.0, us / end_us * 100) if end_us > 0 else 0
                elapsed = time.monotonic() - start
                speed = (us / 1_000_000) / elapsed if elapsed > 0 else 0
                _safe_progress(on_progress, info.path.name, pct, speed)
        except (OSError, ValueError):
            pass

    stderr_t = threading.Thread(target=_stderr_reader, daemon=True)
    stdout_t = threading.Thread(target=_stdout_reader, daemon=True)
    stderr_t.start()
    stdout_t.start()

    rc = -1
    try:
        rc = proc.wait()
    finally:
        _unregister_proc(proc)
        if proc.poll() is None and _cancel_requested.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    stdout_t.join(timeout=3)
    stderr_t.join(timeout=3)
    return rc, ''.join(stderr_lines)


def _cleanup_encode_file(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def convert_video(
    info: VideoInfo,
    opts: ConvertOptions,
    on_progress: Callable[[str, float, float], None],
) -> tuple[bool, str]:
    """Convierte un solo vídeo. on_progress(nombre, pct, speed_x).

    Devuelve (éxito, mensaje). Nunca lanza: errores se devuelven como (False, msg).
    """
    if _cancel_requested.is_set():
        return False, "Cancelado"

    encode_to = info.encode_target(opts.replace_originals)
    final = info.final_path(opts.replace_originals)
    _cleanup_encode_file(encode_to)

    try:
        cmd = _build_cmd(info.path, encode_to, info, opts)
        t0 = time.monotonic()

        rc, err = _run_ffmpeg(cmd, info, on_progress)
        elapsed = time.monotonic() - t0

        if _cancel_requested.is_set():
            _cleanup_encode_file(encode_to)
            return False, "Cancelado"

        if rc != 0:
            tail = err[-400:] if err else ''
            _cleanup_encode_file(encode_to)
            return False, f"ffmpeg exit code {rc}: {tail}"

        if not encode_to.exists():
            return False, "Archivo de salida no encontrado"

        if opts.replace_originals:
            try:
                info.path.unlink()
                encode_to.replace(final)
            except OSError as e:
                _cleanup_encode_file(encode_to)
                return False, f"No se pudo reemplazar el original: {e}"

        out_size = final.stat().st_size
        ratio = out_size / info.size_bytes * 100 if info.size_bytes else 0
        speed_str = (
            f"{info.size_bytes / elapsed / 1_000_000:.1f} MB/s" if elapsed > 0 else "?"
        )
        dest = final.name if opts.replace_originals else f"{final.name} (nuevo)"
        return True, (
            f"{_fmt_size(info.size_bytes)} → {_fmt_size(out_size)} ({ratio:.0f}%) | "
            f"{_fmt_time(elapsed)} | {speed_str} | {dest}"
        )
    except Exception as e:
        _cleanup_encode_file(encode_to)
        return False, f"Error inesperado: {e}"


# ─── UI ─────────────────────────────────────────────────────────────────────

def print_banner():
    gpu = get_gpu_name()
    w = 56
    print()
    print(_box_top(w))
    title = f"{ICO['app']}  {term.bold(term.white('AVCONV'))}"
    print(_box_line(title, w))
    subtitle = (
        f"{term.magenta('Video')} {term.dim('→')} "
        f"{term.cyan('H.264')}  {term.dim('·')}  "
        f"{term.yellow('NVENC')}"
    )
    print(_box_line(subtitle, w))
    print(_box_bot(w))
    print()
    if gpu:
        print(_kv(ICO['gpu'], 'GPU', gpu))
        print()


def print_table(videos: list[VideoInfo], *, av1_only: bool):
    """Muestra tabla de vídeos encontrados."""
    n = len(videos)
    table_w = 2 + _TCOL_NUM + 2 + _TCOL_NAME + _TCOL_RES + _TCOL_DUR + _TCOL_SZ + _TCOL_CODEC
    sep = term.dim('─' * table_w)
    scope = 'AV1' if av1_only else 'a convertir'

    print(sep)
    print(
        f"  {ICO['film']}  {term.bold(term.white(f'{n} vídeos {scope} encontrados'))}"
    )
    print(sep)

    hdr = (
        '  '
        + _cell('#', _TCOL_NUM, 'right', term.dim)
        + '  '
        + _table_row([
            _cell('Nombre', _TCOL_NAME, 'left', lambda t: term.bold(term.cyan(t))),
            _cell('Resolución', _TCOL_RES, 'left', lambda t: term.bold(term.cyan(t))),
            _cell('Duración', _TCOL_DUR, 'left', lambda t: term.bold(term.cyan(t))),
            _cell('Tamaño', _TCOL_SZ, 'left', lambda t: term.bold(term.cyan(t))),
            _cell('Codec', _TCOL_CODEC, 'left', lambda t: term.bold(term.magenta(t))),
        ])
    )
    print(hdr)
    print(sep)

    for i, v in enumerate(videos, 1):
        name = _truncate_display(v.path.name, _TCOL_NAME)
        codec_lbl = _codec_display(v.codec)
        row = (
            '  '
            + _cell(str(i), _TCOL_NUM, 'right', term.dim)
            + '  '
            + _table_row([
                _cell(name, _TCOL_NAME, 'left', term.white),
                _cell(v.resolution, _TCOL_RES, 'left', term.blue),
                _cell(v.dur_str, _TCOL_DUR, 'left', term.dim),
                _cell(v.size_str, _TCOL_SZ, 'left', term.yellow),
                _cell(codec_lbl, _TCOL_CODEC, 'left',
                      lambda t: term.bg_blue(f' {t[:5]} ')),
            ])
        )
        print(row)

    print(sep)
    total_sz = sum(v.size_bytes for v in videos)
    total_dur = sum(v.duration_sec for v in videos)
    print(
        f"  {ICO['stats']}  "
        f"{term.bold(str(n))} {term.dim('vídeos')}  {term.dim('·')}  "
        f"{ICO['size']} {term.yellow(_fmt_size(total_sz))}  {term.dim('·')}  "
        f"{ICO['time']} {term.cyan(str(timedelta(seconds=int(total_dur))))}"
    )
    print(sep)
    print()


def _progress_item_line(name: str, p: dict) -> str:
    """Una línea de progreso para un vídeo."""
    st = p.get('status', 'pending')
    short_name = _truncate_display(name, 44)

    if st == 'done':
        bar = _progress_bar(100)
        return (
            f"  {term.green(ICO['ok'])}  {_cell(short_name, 44, 'left', term.green)}  "
            f"{bar}  {term.green('100%')}  {term.dim('listo')}"
        )
    if st == 'error':
        msg = p.get('message', 'Error')[:32]
        bar = _progress_bar(0)
        return (
            f"  {term.red(ICO['fail'])}  {_cell(short_name, 44, 'left', term.red)}  "
            f"{bar}  {term.red('ERROR')}  {term.dim(msg)}"
        )
    if st == 'running':
        pct = p.get('pct', 0)
        speed = p.get('speed', 0)
        bar = _progress_bar(pct)
        speed_txt = f"{speed:.1f}x" if speed > 0 else "…"
        elapsed_s = p.get('elapsed', 0) or 1
        remaining_s = (100.0 - pct) / max(pct, 1.0) * elapsed_s
        rem_txt = (
            f"  {term.dim('ETA')} {term.cyan(_fmt_time(remaining_s))}"
            if pct > 5 and elapsed_s > 5 else ''
        )
        return (
            f"  {term.cyan(ICO['run'])}  {_cell(short_name, 44, 'left', term.white)}  "
            f"{bar}  {term.bold(term.cyan(f'{pct:5.1f}%'))}  "
            f"{term.yellow(speed_txt)}{rem_txt}"
        )
    bar = _progress_bar(0)
    return (
        f"  {term.dim(ICO['wait'])}  {_cell(short_name, 44, 'left', term.dim)}  "
        f"{bar}  {term.dim('en cola')}"
    )


def _global_progress_pct(progress: dict[str, dict], total: int) -> float:
    done = sum(1 for p in progress.values() if p.get('status') == 'done')
    partial = sum(
        p.get('pct', 0) / 100.0
        for p in progress.values()
        if p.get('status') == 'running'
    )
    return min(100.0, (done + partial) / total * 100) if total else 0.0


def _build_progress_lines_compact(
    progress: dict[str, dict],
    total: int,
    batch_elapsed: float,
) -> list[str]:
    """Vista resumida para muchos vídeos: global + activos + recientes."""
    lines: list[str] = []
    done = sum(1 for p in progress.values() if p.get('status') == 'done')
    errors = [(n, p) for n, p in progress.items() if p.get('status') == 'error']
    running = [(n, p) for n, p in progress.items() if p.get('status') == 'running']
    pending = total - done - len(errors) - len(running)

    pct_global = _global_progress_pct(progress, total)
    global_bar = _progress_bar(pct_global, 36)

    lines.append('')
    lines.append(
        f"  {term.bold(term.white('Conversión en curso'))}  "
        f"{term.dim(f'({total} vídeos · vista compacta)')}"
    )
    lines.append('')

    stats = [
        f"{term.bold(f'{done}/{total}')} {term.dim('listos')}",
        f"{term.cyan(str(len(running)))} {term.dim('activos')}",
    ]
    if errors:
        stats.append(f"{term.red(str(len(errors)))} {term.dim('errores')}")
    if pending > 0:
        stats.append(f"{term.dim(str(pending))} en cola")
    lines.append(f"  {ICO['stats']}  {global_bar}  {' · '.join(stats)}")

    if done > 0 and batch_elapsed > 5 and done < total:
        rate = done / batch_elapsed
        eta = _fmt_time((total - done) / rate)
        lines.append(
            f"     {term.dim('ETA lote')} {term.cyan(eta)}  "
            f"{term.dim('·')}  {ICO['time']} {term.cyan(_fmt_time(batch_elapsed))} transcurrido"
        )
    lines.append('')

    if running:
        lines.append(f"  {term.bold(term.white('En curso'))}")
        for name, p in running:
            lines.append(_progress_item_line(name, p))
        lines.append('')

    recent_done = sorted(
        ((n, p) for n, p in progress.items() if p.get('status') == 'done'),
        key=lambda x: x[1].get('done_at', 0),
        reverse=True,
    )[:COMPACT_RECENT_DONE]
    if recent_done:
        lines.append(f"  {term.bold(term.white('Completados recientes'))}")
        for name, p in recent_done:
            lines.append(_progress_item_line(name, p))
        if done > len(recent_done):
            lines.append(
                f"     {term.dim(f'… y {done - len(recent_done)} más')}"
            )
        lines.append('')

    if errors:
        lines.append(f"  {term.bold(term.red('Errores'))}")
        for name, p in errors[:COMPACT_MAX_ERRORS]:
            lines.append(_progress_item_line(name, p))
        if len(errors) > COMPACT_MAX_ERRORS:
            lines.append(
                f"     {term.dim(f'… y {len(errors) - COMPACT_MAX_ERRORS} errores más')}"
            )
        lines.append('')

    if pending > 0 and not running:
        lines.append(
            f"  {term.dim(ICO['wait'])}  {term.dim(f'{pending} vídeos esperando turno')}"
        )

    return lines


def _build_progress_lines(
    progress: dict[str, dict], total: int, *, batch_elapsed: float = 0.0,
) -> list[str]:
    """Construye líneas de progreso para todos los vídeos."""
    if total > COMPACT_PROGRESS_AT:
        return _build_progress_lines_compact(progress, total, batch_elapsed)

    lines: list[str] = []
    done = 0

    lines.append('')
    lines.append(
        f"  {term.bold(term.white('Conversión en curso'))}  "
        f"{term.dim('─' * 40)}"
    )
    lines.append('')

    for name, p in progress.items():
        if p.get('status') == 'done':
            done += 1
        lines.append(_progress_item_line(name, p))

    pending = total - done
    pct_global = _global_progress_pct(progress, total)
    global_bar = _progress_bar(pct_global, 30)

    lines.append('')
    lines.append(
        f"  {ICO['stats']}  {global_bar}  "
        f"{term.bold(f'{done}/{total}')} {term.dim('completados')}"
    )
    if pending:
        lines.append(f"     {term.dim(f'{pending} pendiente(s)')}")
    return lines


def print_summary(
    results: dict[str, tuple[bool, str]],
    videos: list[VideoInfo],
    ok_count: int,
    fail_count: int,
    elapsed_total: float,
    *,
    replace_originals: bool,
) -> None:
    w = 56
    print()
    print(_box_top(w))
    print(_box_line(f"{ICO['stats']}  {term.bold(term.white('RESUMEN'))}", w))
    print(_box_bot(w))
    print()

    for name, (ok, msg) in sorted(results.items()):
        icon = term.green(ICO['ok']) if ok else term.red(ICO['fail'])
        fname = term.green(name) if ok else term.red(name)
        print(f"  {icon}  {fname}")
        if ok:
            print(f"      {term.dim(msg)}")
        else:
            short = msg[:70] if len(msg) > 70 else msg
            print(f"      {term.red(short)}")

    print()
    ok_part = term.green(f"{ok_count} correctos")
    fail_part = (
        term.red(f"{fail_count} errores") if fail_count
        else term.dim("0 errores")
    )
    print(
        f"  {term.bold('Resultado:')}  {ok_part}  {term.dim('·')}  "
        f"{fail_part}  {term.dim('·')}  "
        f"{ICO['time']} {term.cyan(_fmt_time(elapsed_total))}"
    )

    total_orig = sum(v.size_bytes for v in videos)
    conv_videos = [v for v in videos if results.get(v.path.name, (False, ''))[0]]
    total_new = sum(
        v.final_path(replace_originals).stat().st_size
        for v in conv_videos
        if v.final_path(replace_originals).exists()
    )
    if total_orig > 0:
        pct = total_new / total_orig * 100
        print(
            f"  {ICO['size']}  "
            f"{term.yellow(_fmt_size(total_orig))} {term.dim('→')} "
            f"{term.green(_fmt_size(total_new))} "
            f"{term.dim(f'({pct:.0f}%)')}"
        )
    print()


def _describe_mode(quality_cq: Optional[int], videos: list[VideoInfo]) -> str:
    if quality_cq is not None:
        return f"calidad CQ {quality_cq} (archivos más grandes)"
    avg_br = sum(v.target_video_bitrate_kbps for v in videos) / len(videos)
    return f"tamaño similar (~{_fmt_bitrate(avg_br)} vídeo)"


def _resolve_compression_mode(args, videos: list[VideoInfo]) -> Optional[int]:
    """None = tamaño similar; int = CQ fijo."""
    if args.size_match:
        return None
    if args.quality is not None:
        return args.quality

    avg_br = sum(v.target_video_bitrate_kbps for v in videos) / len(videos)
    idx = _prompt_choice(
        '¿Cómo quieres comprimir el vídeo?',
        [
            (
                'Tamaño similar al original',
                f'~{_fmt_bitrate(avg_br)} vídeo · recomendado para compatibilidad',
            ),
            (
                'Priorizar calidad',
                f'CQ {DEFAULT_QUALITY} · archivos más grandes que el original',
            ),
        ],
        default=0,
    )
    return DEFAULT_QUALITY if idx == 1 else None


def _resolve_edit_friendly(args) -> bool:
    if args.edit_friendly:
        return True
    if args.no_edit_friendly:
        return False
    return _prompt_yes_no(
        '¿Optimizar para edición? (keyframes cada 1 s, sin B-frames, faststart)',
        default=False,
    )


def _resolve_scan_mode(args) -> bool:
    """True = solo AV1; False = cualquier codec distinto de H.264."""
    if args.all_codecs:
        return False
    if args.av1_only:
        return True
    idx = _prompt_choice(
        '¿Qué vídeos convertir?',
        [
            ('Solo AV1', 'descargas o exports recientes'),
            (
                'Cualquier codec',
                'HEVC, VP9, AV1, MPEG-4… excepto H.264',
            ),
        ],
        default=0,
    )
    return idx == 0


def _resolve_replace_originals(args) -> bool:
    if args.replace_originals:
        return True
    if args.keep_originals:
        return False
    return _prompt_yes_no(
        '¿Sobrescribir los archivos originales? '
        '(sustituye el original por H.264 con el mismo nombre)',
        default=False,
    )


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convierte vídeos a H.264 usando GPU NVIDIA (NVENC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  avconv                          directorio actual
  avconv -d ~/Videos              directorio específico
  avconv --all-codecs             HEVC, VP9, AV1, etc.
  avconv --av1-only               solo AV1
  avconv -j 3                     paralelismo manual
  avconv -y                       sin confirmación
  avconv --quality 23             priorizar calidad (sin preguntar)
  avconv --size-match             tamaño similar (sin preguntar)
  avconv --edit-friendly          optimizar para edición
  avconv --replace-originals      sustituir originales in situ (sin preguntar)
  avconv --keep-originals         crear *_h264.* (sin preguntar)
  avconv --dry-run                solo listar
  avconv -f                       sobrescribir *_h264 existentes""",
    )
    ap.add_argument('-d', '--directory', default='.', help='Directorio a escanear')
    ap.add_argument('--all-codecs', action='store_true',
                    help='Convertir cualquier codec (excepto H.264); sin preguntar')
    ap.add_argument('--av1-only', action='store_true',
                    help='Solo vídeos AV1; sin preguntar (comportamiento habitual)')
    ap.add_argument('-j', '--parallel', type=int, default=0,
                    help='Conversiones en paralelo (0 = auto)')
    ap.add_argument('-y', '--yes', action='store_true',
                    help='Saltar confirmación de sobrescritura y conversión')
    ap.add_argument('--quality', type=int, default=None, metavar='CQ',
                    help='Modo calidad fija CQ 0-51 (sin preguntar)')
    ap.add_argument('--size-match', action='store_true',
                    help='Modo tamaño similar al original (sin preguntar)')
    ap.add_argument('--edit-friendly', action='store_true',
                    help='Optimizar para edición en timeline (sin preguntar)')
    ap.add_argument('--no-edit-friendly', action='store_true',
                    help='No optimizar para edición (sin preguntar)')
    ap.add_argument('--replace-originals', action='store_true',
                    help='Sustituir archivos originales (sin preguntar)')
    ap.add_argument('--keep-originals', action='store_true',
                    help='Conservar originales y crear *_h264.* (sin preguntar)')
    ap.add_argument('--dry-run', action='store_true', help='Solo listar')
    ap.add_argument('-f', '--force', action='store_true',
                    help='Sobrescribir archivos _h264 existentes')
    ap.add_argument('--no-color', action='store_true',
                    help='Desactivar colores (útil para logs o pipes)')
    args = ap.parse_args()

    if args.quality is not None and args.size_match:
        _print_error('Usa solo uno: --quality o --size-match.')
        sys.exit(1)
    if args.edit_friendly and args.no_edit_friendly:
        _print_error('Usa solo uno: --edit-friendly o --no-edit-friendly.')
        sys.exit(1)
    if args.replace_originals and args.keep_originals:
        _print_error('Usa solo uno: --replace-originals o --keep-originals.')
        sys.exit(1)
    if args.all_codecs and args.av1_only:
        _print_error('Usa solo uno: --all-codecs o --av1-only.')
        sys.exit(1)

    term.configure(no_color=args.no_color)

    # ── Pre-flight ────────────────────────────────────────────────────────

    for cmd in ('ffmpeg', 'ffprobe'):
        if not _check_cmd(cmd):
            _print_error(f"'{cmd}' no encontrado en el PATH.")
            sys.exit(1)

    if not check_nvenc():
        _print_error("'h264_nvenc' no disponible en ffmpeg.")
        _print_info(
            "Verifica que los drivers NVIDIA están instalados y actualizados."
        )
        sys.exit(1)

    # ── Scan ──────────────────────────────────────────────────────────────

    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        _print_error(f"'{directory}' no es un directorio válido.")
        sys.exit(1)

    print_banner()

    av1_only = _resolve_scan_mode(args)

    print(_kv(ICO['dir'], 'Directorio', str(directory)))
    print(_kv(
        ICO['scan'], 'Modo',
        'solo AV1' if av1_only else 'cualquier codec ≠ H.264',
    ))
    print()
    print(f"  {ICO['scan']}  {term.cyan('Escaneando vídeos…')}")
    print()

    videos = scan_directory(directory, av1_only=av1_only)

    if not videos:
        print()
        scope = 'AV1' if av1_only else 'convertibles (no H.264)'
        _print_warn(f"No se encontraron vídeos {scope} en {directory}")
        _print_info(
            "Búsqueda recursiva · extensiones: "
            + term.dim(', '.join(sorted(VIDEO_EXTS)))
        )
        print()
        sys.exit(0)

    print_table(videos, av1_only=av1_only)

    if args.dry_run:
        _print_info(term.italic("Dry-run: no se realizará conversión."))
        print()
        sys.exit(0)

    # ── Opciones de conversión (interactivas si no vienen por CLI) ────────

    quality_cq = _resolve_compression_mode(args, videos)
    edit_friendly = _resolve_edit_friendly(args)
    replace_originals = _resolve_replace_originals(args)
    conv_opts = ConvertOptions(
        quality_cq=quality_cq,
        edit_friendly=edit_friendly,
        replace_originals=replace_originals,
    )

    # ── Summary ───────────────────────────────────────────────────────────

    total_size = sum(v.size_bytes for v in videos)

    workers = args.parallel or auto_parallel(videos)
    print()
    print(_kv(ICO['parallel'], 'Paralelo', f"{workers} conversiones"))
    print(_kv(ICO['quality'], 'Compresión', _describe_mode(quality_cq, videos)))
    print(_kv(
        ICO['film'],
        'Edición',
        'optimizado (keyframes 1/s)' if edit_friendly else 'reproducción normal',
    ))
    if replace_originals:
        print(_kv(ICO['out'], 'Salida', 'sustituir originales in situ'))
    else:
        print(_kv(ICO['out'], 'Salida', f"conservar AV1 · crear *{OUTPUT_TAG}.*"))
    print()

    # ── Check existing sidecar outputs ────────────────────────────────────

    if not replace_originals:
        conflicts = [v for v in videos if v.sidecar_path.exists()]
        if conflicts and not args.force:
            print()
            for v in conflicts:
                _print_warn(f"Ya existe: {term.yellow(v.sidecar_path.name)}")
            print()
            try:
                resp = _prompt("¿Sobrescribir archivos *_h264 existentes? [S/n]")
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {term.dim('Abortado.')}")
                sys.exit(0)
            if resp and resp not in ('s', 'si', 'y', 'yes', ''):
                _print_info("Cancelado.")
                sys.exit(0)
            args.force = True

    # ── Confirm ───────────────────────────────────────────────────────────

    if not args.yes:
        if replace_originals:
            _print_warn(
                f"Se eliminarán {len(videos)} archivos originales "
                f"tras convertir correctamente."
            )
            print()
        print(
            f"  {ICO['film']}  Se convertirán "
            f"{term.bold(str(len(videos)))} vídeos "
            f"({term.yellow(_fmt_size(total_size))})"
        )
        print()
        try:
            resp = _prompt("¿Convertir todos a H.264? [S/n]")
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {term.dim('Abortado.')}")
            sys.exit(0)
        if resp and resp not in ('s', 'si', 'y', 'yes', ''):
            _print_info("Cancelado.")
            sys.exit(0)

    # ── Progress data ─────────────────────────────────────────────────────

    progress: dict[str, dict] = {}
    for v in videos:
        progress[v.path.name] = {'status': 'pending'}
    progress_lock = threading.Lock()

    results: dict[str, tuple[bool, str]] = {}
    results_lock = threading.Lock()

    def on_progress(name: str, pct: float, speed: float):
        now = time.monotonic()
        with progress_lock:
            d = progress[name]
            if d.get('status') != 'running':
                d['start_time'] = now
            d['status'] = 'running'
            d['pct'] = pct
            d['speed'] = speed
            d['elapsed'] = now - d.get('start_time', now)

    def mark_done(name: str, success: bool, msg: str):
        with progress_lock:
            progress[name] = {
                'status': 'done' if success else 'error',
                'pct': 100.0 if success else 0.0,
                'message': msg,
                'done_at': time.monotonic(),
            }
        with results_lock:
            results[name] = (success, msg)

    # ── Display (clear-screen refresh) ──────────────────────────────────

    stop_display = threading.Event()
    last_pct = {v.path.name: -1 for v in videos}  # solo refrescar si cambia
    batch_started_at = time.monotonic()

    def display_once(first: bool = False, force: bool = False):
        """Imprime el progreso actual. Limpia pantalla salvo en first."""
        try:
            elapsed = time.monotonic() - batch_started_at
            with progress_lock:
                lines = _build_progress_lines(
                    progress, len(videos), batch_elapsed=elapsed,
                )
                changed = force or any(
                    abs(last_pct.get(name, -1) - (p.get('pct', 0) or 0)) > 0.5
                    for name, p in progress.items()
                    if p.get('status') in ('running', 'pending')
                ) or any(
                    p.get('status') in ('done', 'error')
                    for name, p in progress.items()
                    if last_pct.get(name, -1) < 100
                )
                if not changed:
                    return
                for name, p in progress.items():
                    st = p.get('status', 'pending')
                    if st == 'done':
                        last_pct[name] = 100.0
                    elif st == 'error':
                        last_pct[name] = -2.0
                    else:
                        last_pct[name] = p.get('pct', 0) or 0
            if not first:
                _clear_screen()
            print('\n'.join(lines), flush=True)
        except Exception:
            pass

    def display_loop():
        """Refresca cada 1.5s mientras dure la conversión."""
        display_once(first=True)
        while not stop_display.is_set():
            display_once()
            stop_display.wait(1.5)
        display_once(force=True)  # mostrar estado final

    disp = threading.Thread(target=display_loop, daemon=True)
    disp.start()

    # ── Convert ───────────────────────────────────────────────────────────

    signal.signal(signal.SIGINT, _handle_interrupt)

    print()  # separador antes del progreso
    started_at = batch_started_at

    def worker(v: VideoInfo) -> bool:
        if _cancel_requested.is_set():
            mark_done(v.path.name, False, "Cancelado")
            return False
        try:
            _safe_progress(on_progress, v.path.name, 0.0, 0.0)
            ok, msg = convert_video(v, conv_opts, on_progress)
            mark_done(v.path.name, ok, msg)
            return ok
        except Exception as e:
            mark_done(v.path.name, False, f"Error inesperado: {e}")
            return False

    ok_count = 0
    fail_count = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(worker, v): v for v in videos}
            for fut in concurrent.futures.as_completed(futs):
                if _cancel_requested.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    if fut.result():
                        ok_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    v = futs[fut]
                    mark_done(v.path.name, False, f"Error inesperado: {e}")
                    fail_count += 1
    except KeyboardInterrupt:
        _cancel_requested.set()
        terminate_all_procs()
        print(f"\n\n  {term.yellow(ICO['warn'])}  {term.yellow('Interrumpido por el usuario.')}")
        stop_display.set()
        disp.join(timeout=1)
        sys.exit(1)

    if _cancel_requested.is_set():
        stop_display.set()
        disp.join(timeout=1)
        _print_warn("Conversión cancelada.")
        sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────

    stop_display.set()
    disp.join(timeout=1)

    elapsed_total = time.monotonic() - started_at
    print_summary(
        results, videos, ok_count, fail_count, elapsed_total,
        replace_originals=conv_opts.replace_originals,
    )

    if fail_count:
        sys.exit(1)


if __name__ == '__main__':
    main()