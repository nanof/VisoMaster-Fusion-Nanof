"""
AMD FidelityFX Super Resolution 1.x (EASU + RCAS) for video preview (QGraphicsScene + QOpenGLWidget).

Shaders: app/ui/widgets/shaders/fsr1_easu.frag, fsr1_rcas.frag (MIT, AMD copyright in-file).
Preview only; recording/export unchanged.

Por defecto casi no hay salida en consola: tras un frame compuesto bien se imprime una línea
[FSR1] con el pipeline efectivo (y otra vez si cambia modo, p. ej. al quitar ISOLATE).
El resto de trazas FSR1 requieren VISIOMASTER_DEBUG_FSR1 (abajo).

Efecto visual: EASU escala al tamaño del recuadro del preview en la vista. Si ese tamaño
en píxeles es similar al del vídeo (720p cabe en ventana pequeña), el cambio es sutil;
sube el zoom o maximiza la ventana para notar upscale + RCAS (afina «RCAS sharpness»).

Debug (stderr/terminal):
  VISIOMASTER_DEBUG_FSR1=1 — resumen / primer paint OK
  VISIOMASTER_DEBUG_FSR1=2 — además contador de paints
  VISIOMASTER_DEBUG_FSR1=3 — bajo nivel: FBO binding, glGetError tras pasos, uniform locations
  VISIOMASTER_DEBUG_FSR1_GL=1 — igual que nivel 3 sin cambiar el número principal
  VISIOMASTER_GL_GETERROR=1 — volcar glGetError con nombre simbólico (GL_INVALID_VALUE, …) aunque FSR1_GL no esté
  VISIOMASTER_GL_DEBUG=1 — contexto OpenGL con DebugContext + glDebugMessageCallback (mensajes del driver;
      reiniciar la app). El registro del callback se aplaza al siguiente tick del event loop (evita cuelgues
      si se activa durante paint del preview). NOTIFICATION omitidas salvo VISIOMASTER_GL_DEBUG_NOTIFICATIONS=1;
      GL_DEBUG_OUTPUT_SYNCHRONOUS solo con VISIOMASTER_GL_DEBUG_SYNC=1.

Aislar fallos — **anulan el toggle de UI «FSR1: EASU+RCAS shaders»** mientras estén definidas:
  VISIOMASTER_FSR1_ISOLATE=blit_src  — solo textura + blit (nunca EASU ni RCAS)
  VISIOMASTER_FSR1_ISOLATE=easu_only — EASU → FBO → blit (sin paso RCAS)

Windows CMD:  set VISIOMASTER_DEBUG_FSR1=3
PowerShell:   $env:VISIOMASTER_DEBUG_FSR1="3"; $env:VISIOMASTER_FSR1_ISOLATE="blit_src"
If the app runs without a console (pythonw / some shortcuts), prints may not be visible; run python from a terminal.
"""

from __future__ import annotations

import ctypes
from array import array
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from PySide6.QtOpenGL import (
        QOpenGLBuffer,
        QOpenGLFramebufferObject,
        QOpenGLFramebufferObjectFormat,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLTexture,
        QOpenGLVertexArrayObject,
    )
except ImportError:  # pragma: no cover
    QOpenGLBuffer = None  # type: ignore[misc, assignment]
    QOpenGLFramebufferObject = None  # type: ignore[misc, assignment]
    QOpenGLFramebufferObjectFormat = None  # type: ignore[misc, assignment]
    QOpenGLShader = None  # type: ignore[misc, assignment]
    QOpenGLShaderProgram = None  # type: ignore[misc, assignment]
    QOpenGLTexture = None  # type: ignore[misc, assignment]
    QOpenGLVertexArrayObject = None  # type: ignore[misc, assignment]

_GL_FLOAT = 0x1406
_GL_TRIANGLES = 0x0004
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2
_GL_TEXTURE_2D = 0x0DE1
_GL_TEXTURE_MIN_FILTER = 0x2801
_GL_TEXTURE_MAG_FILTER = 0x2800
_GL_LINEAR = 0x2601
_GL_RGBA = 0x1908
_GL_UNSIGNED_BYTE = 0x1401
_GL_RGBA8 = 0x8058
_GL_RGBA32F = 0x8814
_GL_TEXTURE_WRAP_S = 0x2802
_GL_TEXTURE_WRAP_T = 0x2803
_GL_CLAMP_TO_EDGE = 0x812F
_GL_TEXTURE0 = 0x84C0
_GL_SCISSOR_TEST = 0x0C11
_GL_CULL_FACE = 0x0B44
# QOpenGLWidget draws to defaultFramebufferObject(), not the system default (FBO 0).
_GL_FRAMEBUFFER = 0x8D40
_GL_FRAMEBUFFER_BINDING = 0x8CA6
_GL_UNPACK_ALIGNMENT = 0x0CF5
_GL_NO_ERROR = 0
_GL_INVALID_ENUM = 0x0500
_GL_INVALID_VALUE = 0x0501
_GL_INVALID_OPERATION = 0x0502
_GL_STACK_OVERFLOW = 0x0503
_GL_STACK_UNDERFLOW = 0x0504
_GL_OUT_OF_MEMORY = 0x0505
_GL_INVALID_FRAMEBUFFER_OPERATION = 0x0506
_GL_CONTEXT_LOST = 0x0507
# KHR_debug
_GL_DEBUG_OUTPUT = 0x92E0
_GL_DEBUG_OUTPUT_SYNCHRONOUS = 0x8242
_GL_DEBUG_SEVERITY_NOTIFICATION = 0x826B

_GL_ERROR_NAMES: dict[int, str] = {
    _GL_NO_ERROR: "GL_NO_ERROR",
    _GL_INVALID_ENUM: "GL_INVALID_ENUM",
    _GL_INVALID_VALUE: "GL_INVALID_VALUE",
    _GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
    _GL_STACK_OVERFLOW: "GL_STACK_OVERFLOW",
    _GL_STACK_UNDERFLOW: "GL_STACK_UNDERFLOW",
    _GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
    _GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
    _GL_CONTEXT_LOST: "GL_CONTEXT_LOST",
}

_FSR_PROG_VER = 7

# KHR_debug: mantener referencias vivas (GC)
_fsr1_khr_debug_callback_refs: list[Any] = []
_fsr1_khr_debug_installed_ctx_ids: set[int] = set()
_fsr1_gl_geterror_note_shown = False
_gl_driver_logged_ctx: set[int] = set()
_khr_notif_emitted = 0
_khr_notif_cap_warned = False

# Full-viewport NDC quad for EASU/RCAS (they use gl_FragCoord vs u_outSize; must fill the FBO).
_FS_FULLSCREEN_NDC_UV = np.array(
    [
        -1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)


def _fsr1_debug_level() -> int:
    v = os.environ.get("VISIOMASTER_DEBUG_FSR1", "").strip().lower()
    if v in ("3", "gl"):
        return 3
    if v in ("2", "verbose", "all"):
        return 2
    if v in ("1", "true", "yes", "on"):
        return 1
    return 0


def _fsr1_debug_env() -> bool:
    return _fsr1_debug_level() >= 1


def _fsr1_debug_gl() -> bool:
    if os.environ.get("VISIOMASTER_DEBUG_FSR1_GL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    return _fsr1_debug_level() >= 3


def _fsr1_env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on", "all")


def _debug_nis_env() -> bool:
    v = os.environ.get("VISIOMASTER_DEBUG_NIS", "").strip().lower()
    return v in ("1", "true", "yes", "on", "all")


def _fsr1_log_gl_geterrors() -> bool:
    """Si True, drenar y mostrar la cola glGetError (nombres + hex)."""
    return (
        _fsr1_debug_gl()
        or _fsr1_env_truthy("VISIOMASTER_GL_GETERROR")
        or _fsr1_env_truthy("VISIOMASTER_GL_DEBUG")
        or _debug_nis_env()
    )


def _khr_notification_cap() -> int:
    """
    NOTIFICATION sin VISIOMASTER_GL_DEBUG_NOTIFICATIONS=1:
    0 = ninguna; -1 = sin tope; otro entero = máximo a imprimir (defecto 40).
    """
    if _fsr1_env_truthy("VISIOMASTER_GL_DEBUG_NOTIFICATIONS"):
        return -1
    raw = os.environ.get("VISIOMASTER_GL_DEBUG_NOTIFICATION_CAP", "40").strip().lower()
    if raw in ("0", "off", "none"):
        return 0
    try:
        return int(raw)
    except ValueError:
        return 40


def _gl_error_label(code: int) -> str:
    name = _GL_ERROR_NAMES.get(int(code))
    if name is not None:
        return f"{name} ({hex(int(code))})"
    return hex(int(code))


def _fsr1_try_install_khr_debug(ctx: QtGui.QOpenGLContext | None) -> None:
    """
    Registra glDebugMessageCallback si existe y VISIOMASTER_GL_DEBUG=1.
    Requiere contexto con DebugContext (preview_opengl_surface_format) para muchos drivers.
    """
    if ctx is None or not _fsr1_env_truthy("VISIOMASTER_GL_DEBUG"):
        return
    cid = id(ctx)
    if cid in _fsr1_khr_debug_installed_ctx_ids:
        return
    try:
        addr = int(ctx.getProcAddress(b"glDebugMessageCallback"))
    except Exception:
        addr = 0
    if addr == 0:
        try:
            addr = int(ctx.getProcAddress(b"glDebugMessageCallbackKHR"))
        except Exception:
            addr = 0
    if addr == 0:
        _msg_cb_missing = (
            "VISIOMASTER_GL_DEBUG: glDebugMessageCallback no encontrado "
            "(¿OpenGL < 4.3 / sin KHR_debug?)"
        )
        if _fsr1_log_gl_geterrors() and _fsr1_debug_env():
            _fsr1_dbg(_msg_cb_missing)
        elif _fsr1_env_truthy("VISIOMASTER_GL_DEBUG"):
            print(f"[GL] {_msg_cb_missing}", flush=True)
        _fsr1_khr_debug_installed_ctx_ids.add(cid)
        return

    functype = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
    DEBUGPROC = functype(
        None,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    SET_CALLBACK = functype(None, ctypes.c_void_p, ctypes.c_void_p)

    def _on_debug(source, type_, id_, severity, length, message, _user) -> None:
        global _khr_notif_emitted, _khr_notif_cap_warned
        sev_int = int(severity)
        if sev_int == _GL_DEBUG_SEVERITY_NOTIFICATION:
            cap = _khr_notification_cap()
            if cap == 0:
                return
            if cap > 0 and _khr_notif_emitted >= cap:
                if not _khr_notif_cap_warned:
                    _khr_notif_cap_warned = True
                    print(
                        "[GL KHR_debug] (NOTIFICATION: límite alcanzado; "
                        "VISIOMASTER_GL_DEBUG_NOTIFICATIONS=1 para todas, "
                        "o sube VISIOMASTER_GL_DEBUG_NOTIFICATION_CAP)",
                        flush=True,
                    )
                return
            if cap > 0:
                _khr_notif_emitted += 1
        try:
            if message:
                if length and int(length) > 0:
                    text = ctypes.string_at(message, int(length)).decode(
                        "utf-8", errors="replace"
                    )
                else:
                    text = ctypes.string_at(message).decode("utf-8", errors="replace")
            else:
                text = ""
        except Exception:
            text = "<?>"
        sev_n = _GL_DEBUG_SEVERITY_NAMES.get(int(severity), hex(int(severity)))
        typ_n = _GL_DEBUG_TYPE_NAMES.get(int(type_), hex(int(type_)))
        print(
            f"[GL KHR_debug] severity={sev_n} type={typ_n} id={int(id_)}: {text}",
            flush=True,
        )

    cb = DEBUGPROC(_on_debug)
    _fsr1_khr_debug_callback_refs.append(cb)
    try:
        setter = SET_CALLBACK(addr)
        setter(ctypes.cast(cb, ctypes.c_void_p), None)
    except Exception as e:
        _gl_debug_setup_log(
            f"VISIOMASTER_GL_DEBUG: glDebugMessageCallback falló: {e!r}"
        )
        _fsr1_khr_debug_installed_ctx_ids.add(cid)
        return

    xf = _gl_extra_for_context(ctx)
    try:
        if hasattr(xf, "glEnable"):
            xf.glEnable(_GL_DEBUG_OUTPUT)
            if _fsr1_env_truthy("VISIOMASTER_GL_DEBUG_SYNC"):
                xf.glEnable(_GL_DEBUG_OUTPUT_SYNCHRONOUS)
    except Exception:
        pass
    _fsr1_khr_debug_installed_ctx_ids.add(cid)
    _gl_debug_setup_log(
        "VISIOMASTER_GL_DEBUG: callback KHR_debug instalado "
        "(NOTIFICATION con tope por defecto; sync solo con VISIOMASTER_GL_DEBUG_SYNC=1)"
    )
    if _fsr1_env_truthy("VISIOMASTER_GL_DEBUG"):
        cap = _khr_notification_cap()
        if cap < 0:
            cap_hint = "todas (VISIOMASTER_GL_DEBUG_NOTIFICATIONS=1)"
        elif cap == 0:
            cap_hint = (
                "ninguna (¿VISIOMASTER_GL_DEBUG_NOTIFICATION_CAP=0? sube el valor o "
                "usa VISIOMASTER_GL_DEBUG_NOTIFICATIONS=1)"
            )
        else:
            cap_hint = f"primeras {cap} (VISIOMASTER_GL_DEBUG_NOTIFICATION_CAP)"
        print(
            f"[GL] NOTIFICATION: {cap_hint}. "
            "NIS: VISIOMASTER_DEBUG_NIS=1 + glGetError con GL_DEBUG o DEBUG_NIS.",
            flush=True,
        )


def schedule_khr_debug_install_for_context(ctx: QtGui.QOpenGLContext | None) -> None:
    """
    No registrar glDebugMessageCallback ni glEnable(DEBUG_OUTPUT) en el mismo stack que
    QGraphicsView.paintEvent sobre QOpenGLWidget: en Windows + algunos drivers la UI puede
    quedar colgada justo después del log «callback instalado». Se difiere al siguiente tick.
    """
    if ctx is None or not _fsr1_env_truthy("VISIOMASTER_GL_DEBUG"):
        return
    if id(ctx) in _fsr1_khr_debug_installed_ctx_ids:
        return
    app = QtCore.QCoreApplication.instance()
    if app is None:
        _fsr1_try_install_khr_debug(ctx)
        return
    QtCore.QTimer.singleShot(0, lambda c=ctx: _fsr1_try_install_khr_debug(c))


_GL_DEBUG_SEVERITY_NAMES: dict[int, str] = {
    0x9146: "HIGH",
    0x9147: "MEDIUM",
    0x9148: "LOW",
    0x826B: "NOTIFICATION",
}

_GL_DEBUG_TYPE_NAMES: dict[int, str] = {
    0x824C: "ERROR",
    0x824D: "DEPRECATED_BEHAVIOR",
    0x824E: "UNDEFINED_BEHAVIOR",
    0x824F: "PORTABILITY",
    0x8250: "PERFORMANCE",
    0x8268: "MARKER",
    0x8269: "PUSH_GROUP",
    0x826A: "POP_GROUP",
    0x826B: "OTHER",
}


def _env_strip_outer_quotes(raw: str) -> str:
    """CMD / env files sometimes store values as \"blit_src\" with literal quotes."""
    s = raw.strip()
    if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        return s[1:-1].strip()
    return s


def _fsr1_isolate_mode() -> str:
    return _env_strip_outer_quotes(
        os.environ.get("VISIOMASTER_FSR1_ISOLATE", "")
    ).lower()


def _fsr1_dbg(msg: str) -> None:
    if _fsr1_debug_env():
        print(f"[FSR1] {msg}", flush=True)


def _gl_debug_setup_log(msg: str) -> None:
    """VISIOMASTER_GL_DEBUG sin FSR1 debug: prefijo [GL]; con FSR1 debug reutiliza _fsr1_dbg."""
    if _fsr1_debug_env():
        _fsr1_dbg(msg)
    elif _fsr1_env_truthy("VISIOMASTER_GL_DEBUG"):
        print(f"[GL] {msg}", flush=True)


def _gl_opengl_context_for_widget(
    gl_widget: QtWidgets.QWidget | None, ctx: QtGui.QOpenGLContext | None
) -> QtGui.QOpenGLContext | None:
    if gl_widget is not None:
        gc = getattr(gl_widget, "context", None)
        if callable(gc):
            try:
                wctx = gc()
                if wctx is not None:
                    return wctx
            except Exception:
                pass
    if ctx is not None:
        return ctx
    return QtGui.QOpenGLContext.currentContext()


def _gl_extra_for_context(octx: QtGui.QOpenGLContext) -> Any:
    """QOpenGLExtraFunctions(context)+init: PySide6 a veces no resuelve FBO con ctx.extraFunctions() solo."""
    from PySide6.QtGui import QOpenGLExtraFunctions

    xf = QOpenGLExtraFunctions(octx)
    if hasattr(xf, "initializeOpenGLFunctions"):
        try:
            xf.initializeOpenGLFunctions()
        except Exception:
            pass
    return xf


def _gl_base_functions(ctx: QtGui.QOpenGLContext) -> Any:
    """ctx.functions() + init (glViewport, glActiveTexture, …)."""
    f = ctx.functions()
    if f is not None and hasattr(f, "initializeOpenGLFunctions"):
        try:
            f.initializeOpenGLFunctions()
        except Exception:
            pass
    return f


def log_gl_driver_info_once(ctx: QtGui.QOpenGLContext | None) -> None:
    """Una vez por contexto: vendor/renderer/version/glsl (no depende de KHR NOTIFICATION)."""
    if ctx is None:
        return
    if not (
        _fsr1_env_truthy("VISIOMASTER_GL_DEBUG")
        or _debug_nis_env()
    ):
        return
    cid = id(ctx)
    if cid in _gl_driver_logged_ctx:
        return
    _gl_driver_logged_ctx.add(cid)
    _GL_VENDOR = 0x1F00
    _GL_RENDERER = 0x1F01
    _GL_VERSION = 0x1F02
    _GL_GLSL = 0x8B8C

    def _decode_gs(raw: object) -> str:
        if raw is None:
            return "?"
        try:
            if isinstance(raw, (bytes, bytearray)):
                return raw.decode("utf-8", errors="replace")
            if hasattr(raw, "data") and callable(raw.data):
                return bytes(raw.data()).decode("utf-8", errors="replace")
            return str(raw)
        except Exception:
            return "?"

    try:
        bf = ctx.functions()
        if bf is not None and hasattr(bf, "initializeOpenGLFunctions"):
            try:
                bf.initializeOpenGLFunctions()
            except Exception:
                pass
        if bf is None or not hasattr(bf, "glGetString"):
            print("[GL] glGetString no disponible en ctx.functions()", flush=True)
            return
        for label, enum in (
            ("vendor", _GL_VENDOR),
            ("renderer", _GL_RENDERER),
            ("version", _GL_VERSION),
            ("glsl", _GL_GLSL),
        ):
            try:
                s = _decode_gs(bf.glGetString(enum))
            except Exception as exc:
                s = f"<?> ({exc!r})"
            print(f"[GL] {label}={s}", flush=True)
    except Exception as e:
        print(f"[GL] driver info error: {e!r}", flush=True)


def gl_texture_barrier_after_image_write(ctx: QtGui.QOpenGLContext | None) -> None:
    """OpenGL 4.5: coherencia imageStore → muestreo como textura (evita negro en algunos drivers)."""
    if ctx is None:
        return
    xf = _gl_extra_for_context(ctx)
    tb = getattr(xf, "glTextureBarrier", None)
    if callable(tb):
        try:
            tb()
        except Exception:
            pass


def _gl_texture_functions(ctx: QtGui.QOpenGLContext) -> Any:
    """
    Funciones para glGenTextures/glTexImage2D. ctx.functions() a veces no expone glGenTextures en PySide6;
    QOpenGLVersionFunctionsFactory.get(perfil del contexto) sí.
    """
    f = ctx.functions()
    if f is not None and hasattr(f, "initializeOpenGLFunctions"):
        try:
            f.initializeOpenGLFunctions()
        except Exception:
            pass
    chosen = f
    if f is None or not hasattr(f, "glGenTextures"):
        try:
            from PySide6.QtOpenGL import (
                QOpenGLVersionFunctionsFactory,
                QOpenGLVersionProfile,
            )

            fmt = ctx.format()
            vp = QOpenGLVersionProfile()
            vp.setVersion(int(fmt.majorVersion()), int(fmt.minorVersion()))
            vp.setProfile(fmt.profile())
            vf = QOpenGLVersionFunctionsFactory.get(vp, ctx)
            if vf is not None and hasattr(vf, "initializeOpenGLFunctions"):
                try:
                    vf.initializeOpenGLFunctions()
                except Exception:
                    pass
            if vf is not None and hasattr(vf, "glGenTextures"):
                chosen = vf
        except Exception:
            pass
    return chosen


def _gl_gen_texture_name(f: Any) -> int:
    if f is None or not hasattr(f, "glGenTextures"):
        return 0
    try:
        r = f.glGenTextures(1)
        if isinstance(r, int) and r > 0:
            return r
        try:
            tr = int(r)
            if tr > 0:
                return tr
        except (TypeError, ValueError):
            pass
        if isinstance(r, (list, tuple)) and len(r) >= 1:
            t = int(r[0])
            if t > 0:
                return t
    except TypeError:
        pass
    except Exception:
        pass
    try:
        r = f.glGenTextures()
        if isinstance(r, int) and r > 0:
            return r
    except Exception:
        pass
    try:
        ids = array("I", [0])
        f.glGenTextures(1, ids)
        t = int(ids[0])
        if t > 0:
            return t
    except Exception:
        pass
    try:
        buf = [0]
        f.glGenTextures(1, buf)
        return int(buf[0])
    except Exception:
        pass
    try:
        u = ctypes.c_uint32(0)
        f.glGenTextures(1, ctypes.byref(u))
        return int(u.value)
    except Exception:
        pass
    try:
        arr = (ctypes.c_uint32 * 1)()
        f.glGenTextures(1, arr)
        return int(arr[0])
    except Exception:
        return 0


def _gl_delete_texture_name(f: Any, tid: int) -> None:
    if tid <= 0 or f is None or not hasattr(f, "glDeleteTextures"):
        return
    try:
        a = array("I", [int(tid)])
        f.glDeleteTextures(1, a)
        return
    except Exception:
        pass
    try:
        buf = [int(tid)]
        f.glDeleteTextures(1, buf)
    except Exception:
        try:
            arr = (ctypes.c_uint32 * 1)(int(tid))
            f.glDeleteTextures(1, arr)
        except Exception:
            pass


def _gl_ctypes_fn():
    return ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE


def _fsr_gl_c_proc_addr(ctx: QtGui.QOpenGLContext, name: bytes) -> int:
    try:
        return int(ctx.getProcAddress(name))
    except Exception:
        return 0


def _fsr_gl_c_gen_textures(ctx: QtGui.QOpenGLContext) -> int:
    p = _fsr_gl_c_proc_addr(ctx, b"glGenTextures")
    if p == 0:
        return 0
    Fn = _gl_ctypes_fn()(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32))(p)
    buf = (ctypes.c_uint32 * 1)()
    Fn(1, buf)
    t = int(buf[0])
    return t if t > 0 else 0


def _fsr_gl_c_delete_textures(ctx: QtGui.QOpenGLContext, tid: int) -> None:
    if tid <= 0:
        return
    p = _fsr_gl_c_proc_addr(ctx, b"glDeleteTextures")
    if p == 0:
        return
    Fn = _gl_ctypes_fn()(None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32))(p)
    buf = (ctypes.c_uint32 * 1)(int(tid))
    Fn(1, buf)


def _fsr_gl_c_active_texture(ctx: QtGui.QOpenGLContext, texture_unit_enum: int) -> None:
    p = _fsr_gl_c_proc_addr(ctx, b"glActiveTexture")
    if p == 0:
        return
    Fn = _gl_ctypes_fn()(None, ctypes.c_uint32)(p)
    Fn(int(texture_unit_enum))


def _fsr_gl_c_bind_texture(ctx: QtGui.QOpenGLContext, target: int, tid: int) -> None:
    p = _fsr_gl_c_proc_addr(ctx, b"glBindTexture")
    if p == 0:
        return
    Fn = _gl_ctypes_fn()(None, ctypes.c_uint32, ctypes.c_uint32)(p)
    Fn(int(target), int(tid))


def _fsr_gl_c_tex_parameter_i(
    ctx: QtGui.QOpenGLContext, target: int, pname: int, param: int
) -> None:
    p = _fsr_gl_c_proc_addr(ctx, b"glTexParameteri")
    if p == 0:
        return
    Fn = _gl_ctypes_fn()(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32)(p)
    Fn(int(target), int(pname), int(param))


def _fsr_gl_c_pixel_store_i(ctx: QtGui.QOpenGLContext, pname: int, param: int) -> None:
    p = _fsr_gl_c_proc_addr(ctx, b"glPixelStorei")
    if p == 0:
        return
    Fn = _gl_ctypes_fn()(None, ctypes.c_uint32, ctypes.c_int32)(p)
    Fn(int(pname), int(param))


def _fsr_gl_c_tex_image_2d_rgba8_empty(
    ctx: QtGui.QOpenGLContext, target: int, w: int, h: int
) -> bool:
    p = _fsr_gl_c_proc_addr(ctx, b"glTexImage2D")
    if p == 0:
        return False
    Fn = _gl_ctypes_fn()(
        None,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )(p)
    try:
        Fn(
            int(target),
            0,
            int(_GL_RGBA8),
            int(w),
            int(h),
            0,
            int(_GL_RGBA),
            int(_GL_UNSIGNED_BYTE),
            None,
        )
    except Exception:
        Fn(
            int(target),
            0,
            int(_GL_RGBA8),
            int(w),
            int(h),
            0,
            int(_GL_RGBA),
            int(_GL_UNSIGNED_BYTE),
            ctypes.c_void_p(0),
        )
    return True


def _fsr_gl_c_tex_image_2d_rgba32f(
    ctx: QtGui.QOpenGLContext, target: int, w: int, h: int, pixels_addr: int
) -> bool:
    """Subida coef NIS (RGBA32F); usa el mismo camino ctypes que la textura vídeo raw."""
    p = _fsr_gl_c_proc_addr(ctx, b"glTexImage2D")
    if p == 0:
        return False
    Fn = _gl_ctypes_fn()(
        None,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )(p)
    try:
        Fn(
            int(target),
            0,
            int(_GL_RGBA32F),
            int(w),
            int(h),
            0,
            int(_GL_RGBA),
            int(_GL_FLOAT),
            ctypes.c_void_p(int(pixels_addr)),
        )
    except Exception:
        return False
    return True


def _fsr_gl_c_tex_sub_image_2d_rgba(
    ctx: QtGui.QOpenGLContext, target: int, w: int, h: int, rgba: np.ndarray
) -> None:
    p = _fsr_gl_c_proc_addr(ctx, b"glTexSubImage2D")
    if p == 0:
        return
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    if not arr.flags.writeable:
        arr = arr.copy()
    buf = (ctypes.c_uint8 * arr.size).from_buffer(arr)
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    Fn = _gl_ctypes_fn()(
        None,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )(p)
    Fn(
        int(target),
        0,
        0,
        0,
        int(w),
        int(h),
        int(_GL_RGBA),
        int(_GL_UNSIGNED_BYTE),
        ptr,
    )


def _fsr_gl_active_texture_unit0(ctx: QtGui.QOpenGLContext | None) -> None:
    """Tras QPainter en el viewport, la unidad activa puede no ser TEXTURE0; el sampler u_src usa 0."""
    if ctx is None:
        return
    f = _gl_base_functions(ctx)
    if f is not None and hasattr(f, "glActiveTexture"):
        try:
            f.glActiveTexture(_GL_TEXTURE0)
            return
        except Exception:
            pass
    xf = _gl_extra_for_context(ctx)
    if hasattr(xf, "glActiveTexture"):
        try:
            xf.glActiveTexture(_GL_TEXTURE0)
        except Exception:
            pass


def _gl_viewport_phys_size(gl_widget: QtWidgets.QWidget) -> tuple[int, int]:
    """Tamaño del FBO del QOpenGLWidget (lógico × DPR); debe coincidir con glViewport."""
    try:
        dpr = float(gl_widget.devicePixelRatioF())
    except Exception:
        dpr = 1.0
    if dpr <= 0.0:
        dpr = 1.0
    lw = max(1, int(gl_widget.width()))
    lh = max(1, int(gl_widget.height()))
    return max(1, int(round(lw * dpr))), max(1, int(round(lh * dpr)))


def _gl_bind_framebuffer(
    f: Any,
    ctx: QtGui.QOpenGLContext | None,
    gl_widget: QtWidgets.QWidget | None,
    fid: int,
) -> None:
    if fid <= 0:
        return
    octx = _gl_opengl_context_for_widget(gl_widget, ctx)
    if octx is not None:
        try:
            xf = _gl_extra_for_context(octx)
            if hasattr(xf, "glBindFramebuffer"):
                xf.glBindFramebuffer(_GL_FRAMEBUFFER, int(fid))
                return
        except Exception:
            pass
        try:
            xf2 = octx.extraFunctions()
            if xf2 is not None and hasattr(xf2, "glBindFramebuffer"):
                xf2.glBindFramebuffer(_GL_FRAMEBUFFER, int(fid))
                return
        except Exception:
            pass
    b = getattr(f, "glBindFramebuffer", None)
    if b is not None:
        b(_GL_FRAMEBUFFER, int(fid))


def _gl_query_framebuffer_binding(
    f: Any,
    ctx: QtGui.QOpenGLContext | None,
    gl_widget: QtWidgets.QWidget | None,
) -> int | None:
    octx = _gl_opengl_context_for_widget(gl_widget, ctx)
    if octx is not None:
        try:
            xf = _gl_extra_for_context(octx)
            if hasattr(xf, "glGetIntegerv"):
                buf = [0]
                xf.glGetIntegerv(_GL_FRAMEBUFFER_BINDING, buf)
                return int(buf[0])
        except Exception:
            pass
        try:
            xf2 = octx.extraFunctions()
            if xf2 is not None and hasattr(xf2, "glGetIntegerv"):
                buf = [0]
                xf2.glGetIntegerv(_GL_FRAMEBUFFER_BINDING, buf)
                return int(buf[0])
        except Exception:
            pass
    get_iv = getattr(f, "glGetIntegerv", None)
    if get_iv is None:
        return None
    try:
        buf = [0]
        get_iv(_GL_FRAMEBUFFER_BINDING, buf)
        return int(buf[0])
    except Exception:
        return None


def _fsr1_gl_drain_errors(ctx: QtGui.QOpenGLContext | None, tag: str) -> None:
    """
    Vacía la cola de errores con glGetError (hasta 256 entradas).

    glGetError solo devuelve un código por llamada (p. ej. GL_INVALID_VALUE); la especificación
    no dice *qué* parámetro era inválido. Para texto del driver (función, mensaje), usar
    VISIOMASTER_GL_DEBUG=1 con contexto debug.
    """
    global _fsr1_gl_geterror_note_shown
    if ctx is None or not _fsr1_log_gl_geterrors():
        return
    xf = _gl_extra_for_context(ctx)
    if not hasattr(xf, "glGetError"):
        return
    codes: list[int] = []
    for _ in range(256):
        try:
            e = int(xf.glGetError())
        except Exception:
            break
        if e == _GL_NO_ERROR:
            break
        codes.append(e)
    if not codes:
        return

    def _gl_geterror_log_bracket(t: str) -> str:
        tl = t.lower()
        if "nis" in tl:
            return "NIS"
        if any(
            s in tl
            for s in (
                "easu",
                "rcas",
                "incoming",
                "direct blit",
                "composite",
                "isolate",
            )
        ):
            return "FSR1"
        if _fsr1_debug_env():
            return "FSR1"
        if _debug_nis_env():
            return "NIS"
        return "GL"

    bracket = _gl_geterror_log_bracket(tag)
    lines = [f"{tag} glGetError[{i}] {_gl_error_label(c)}" for i, c in enumerate(codes)]
    for ln in lines:
        if _fsr1_debug_env():
            _fsr1_dbg(ln)
        else:
            print(f"[{bracket}] {ln}", flush=True)
    if not _fsr1_gl_geterror_note_shown:
        _fsr1_gl_geterror_note_shown = True
        note = (
            "Nota: glGetError no indica qué argumento falló; para mensajes detallados del "
            "driver: VISIOMASTER_GL_DEBUG=1 y reinicio (contexto OpenGL con DebugContext)."
        )
        if _fsr1_debug_env():
            _fsr1_dbg(note)
        else:
            print(f"[{bracket}] {note}", flush=True)


def _fsr1_gl_log_fb(ctx: QtGui.QOpenGLContext, gl_widget: QtWidgets.QWidget, tag: str) -> None:
    if not _fsr1_debug_gl():
        return
    f = ctx.functions()
    raw = _gl_query_framebuffer_binding(f, ctx, gl_widget)
    dfo: int | str = "?"
    try:
        dfo = int(gl_widget.defaultFramebufferObject())
    except Exception as e:
        dfo = f"err:{e!r}"
    _fsr1_dbg(f"{tag} FBO_BINDING(raw)={raw} defaultFramebufferObject={dfo}")


_VS = """#version 330 core
layout(location = 0) in vec4 a_pos_uv;
out vec2 v_uv;
void main() {
  gl_Position = vec4(a_pos_uv.xy, 0.0, 1.0);
  v_uv = a_pos_uv.zw;
}
"""

# Single-texture blit: mismas UV que EASU/RCAS en el quad. No usar 1.0-v aquí:
# la textura vídeo se sube con glTex(Sub)Image2D (orden OpenGL: 1ª fila → t bajo);
# el flip extra invertía la imagen frente al preview pixmap / ctypes.
_BLIT_FS = """#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
layout(location = 0) out vec4 fragColor;
void main() {
  fragColor = vec4(texture(u_src, v_uv).rgb, 1.0);
}
"""

_SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


def _load_frag(name: str) -> str:
    p = _SHADERS_DIR / name
    return p.read_text(encoding="utf-8")


def _qgl_uniform_location(program: Any, name: str) -> int:
    if program is None or not name:
        return -1
    for key in (name, name.encode("ascii")):
        try:
            loc = int(program.uniformLocation(key))
        except (TypeError, ValueError):
            continue
        if loc >= 0:
            return loc
    return -1


def _qgl_uniform_location_gl(program: Any, name: str) -> int:
    """Resolve uniform location via GL when QOpenGLShaderProgram.uniformLocation fails."""
    if program is None or not name:
        return -1
    ctx = QtGui.QOpenGLContext.currentContext()
    if ctx is None:
        return -1
    xf = ctx.extraFunctions()
    pid = int(program.programId())
    if xf is None or pid <= 0:
        return -1
    try:
        gloc = int(xf.glGetUniformLocation(pid, name.encode("utf-8")))
    except Exception:
        return -1
    return gloc if gloc >= 0 else -1


def _numpy_bgr_to_rgb_contiguous(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 BGR")
    return np.ascontiguousarray(arr[..., ::-1])


def _numpy_rgb_to_rgba_contiguous(rgb: np.ndarray) -> np.ndarray:
    """RGB8 → RGBA8 (alpha=255). Evita RGB8 textura + setData(RGB) que en algunos drivers da Invalid texture format."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 RGB")
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = 255
    return np.ascontiguousarray(out)


class VideoPreviewFsrGlItem(QtWidgets.QGraphicsObject):
    """FSR1 (EASU → RCAS) or debug blit (texture only) on the OpenGL viewport."""

    def __init__(self) -> None:
        super().__init__()
        self._frame_bgr: np.ndarray | None = None
        self._sharpness: float = 0.35
        self._use_pipeline_shaders: bool = True
        self._lay_w: int = 1
        self._lay_h: int = 1
        self._display_frame: int = -1
        # Monotonic: one increment per set_frame_sharpness (do not dedupe by frame# / ndarray id;
        # workers may reuse the same buffer and decoupled preview may repeat ui frame indices).
        self._present_seq: int = 0
        self._upload_applied_seq: int = -1
        self._prog_easu: QOpenGLShaderProgram | None = None
        self._prog_rcas: QOpenGLShaderProgram | None = None
        self._prog_blit: QOpenGLShaderProgram | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._vbo_fs: QOpenGLBuffer | None = None
        self._vao: Any = None  # QOpenGLVertexArrayObject | None; Core profile needs VAO for glDrawArrays
        # Textura vídeo: GL directo (glTexImage2D), no QOpenGLTexture — evita glTexStorage + setData
        # que en algunos drivers NVIDIA sigue dando "Invalid texture format".
        self._tex_src_gl_id: int = 0
        self._tex_src_gl_w: int = 0
        self._tex_src_gl_h: int = 0
        self._tex_src_qt: Any = None  # QOpenGLTexture | None (fallback)
        self._tex_src_backend: str = "none"  # "raw" | "qt"
        self._tex_src_raw_ctypes: bool = False  # raw vía getProcAddress+ctypes (PySide glGenTextures→0)
        self._fbo: QOpenGLFramebufferObject | None = None
        self._fbo_rcas: QOpenGLFramebufferObject | None = None
        self._fbo_w: int = 0
        self._fbo_h: int = 0
        self._gl_failed: bool = False
        self._vbo_cache_key: tuple[Any, ...] | None = None
        self._cached_verts: np.ndarray | None = None
        self._fsr1_gl_extra_logged: bool = False
        self._fsr1_isolate_logged: bool = False
        self._fsr1_isolate_bad_logged: bool = False
        self._fsr1_pipeline_path_key: str | None = None
        # Last spatial pass resolutions for preview overlay (tex W×H → output W×H).
        self._preview_overlay_src_wh: tuple[int, int] | None = None
        self._preview_overlay_tgt_wh: tuple[int, int] | None = None
        self.setZValue(1.0)
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802
        return QtCore.QRectF(0.0, 0.0, float(self._lay_w), float(self._lay_h))

    def set_frame_sharpness(
        self,
        texture_bgr: np.ndarray,
        sharpness: float,
        *,
        layout_hw: tuple[int, int],
        display_frame: int,
        use_pipeline_shaders: bool = True,
    ) -> None:
        use_pl = bool(use_pipeline_shaders)
        if use_pl != getattr(self, "_use_pipeline_shaders", True):
            self._vbo_cache_key = None
            self._cached_verts = None
        self._use_pipeline_shaders = use_pl
        lh, lw = int(layout_hw[0]), int(layout_hw[1])
        th, tw = texture_bgr.shape[:2]
        if lw != self._lay_w or lh != self._lay_h:
            self.prepareGeometryChange()
            self._lay_w = lw
            self._lay_h = lh
            self._vbo_cache_key = None
            self._cached_verts = None
            self._invalidate_fbo()
        if tw != getattr(self, "_last_tw", -1) or th != getattr(self, "_last_th", -1):
            self._upload_applied_seq = -1
            self._last_tw = int(tw)
            self._last_th = int(th)
        self._frame_bgr = texture_bgr
        self._sharpness = float(max(0.0, min(2.0, sharpness)))
        self._display_frame = int(display_frame)
        self._present_seq += 1
        self.update()

    def _invalidate_fbo(self) -> None:
        if self._fbo is not None:
            self._fbo = None
        if self._fbo_rcas is not None:
            self._fbo_rcas = None
        self._fbo_w = 0
        self._fbo_h = 0

    def _destroy_gl_objects(self) -> None:
        self._invalidate_fbo()
        try:
            self._src_gl_destroy(QtGui.QOpenGLContext.currentContext())
        except Exception:
            self._tex_src_gl_id = 0
            self._tex_src_gl_w = 0
            self._tex_src_gl_h = 0
        if self._vbo is not None:
            self._vbo.destroy()
            self._vbo = None
        if self._vbo_fs is not None:
            self._vbo_fs.destroy()
            self._vbo_fs = None
        if self._vao is not None:
            try:
                self._vao.destroy()
            except Exception:
                pass
            self._vao = None
        for p in (self._prog_easu, self._prog_rcas, self._prog_blit):
            if p is not None:
                p.removeAllShaders()
        self._prog_easu = None
        self._prog_rcas = None
        self._prog_blit = None
        self._upload_applied_seq = -1
        self._vbo_cache_key = None
        self._cached_verts = None

    def _apply_fsr_program_version(self) -> None:
        if getattr(self, "_fsr_prog_ver", 0) == _FSR_PROG_VER:
            return
        for p in (self._prog_easu, self._prog_rcas, self._prog_blit):
            if p is not None:
                p.removeAllShaders()
        self._prog_easu = None
        self._prog_rcas = None
        self._prog_blit = None
        self._fsr_prog_ver = _FSR_PROG_VER

    def _ensure_blit_program(self) -> bool:
        if QOpenGLShaderProgram is None or QOpenGLShader is None:
            return False
        if self._prog_blit is not None and self._prog_blit.isLinked():
            return True
        self._prog_blit = QOpenGLShaderProgram()
        self._prog_blit.bindAttributeLocation("a_pos_uv", 0)
        ok = self._prog_blit.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _VS
        )
        ok = ok and self._prog_blit.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _BLIT_FS
        )
        ok = ok and self._prog_blit.link()
        if not ok:
            if self._prog_blit is not None:
                print("[FSR1] Blit shader link failed:", self._prog_blit.log())
            self._prog_blit.removeAllShaders()
            self._prog_blit = None
            return False
        return True

    def _ensure_fsr_programs(self) -> bool:
        if QOpenGLShaderProgram is None or QOpenGLShader is None:
            return False
        try:
            easu_src = _load_frag("fsr1_easu.frag")
            rcas_src = _load_frag("fsr1_rcas.frag")
        except OSError:
            return False

        if self._prog_easu is None or not self._prog_easu.isLinked():
            self._prog_easu = QOpenGLShaderProgram()
            self._prog_easu.bindAttributeLocation("a_pos_uv", 0)
            ok = self._prog_easu.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, _VS
            )
            ok = ok and self._prog_easu.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, easu_src
            )
            ok = ok and self._prog_easu.link()
            if not ok:
                if self._prog_easu is not None:
                    print(
                        "[FSR1] EASU shader link failed:",
                        self._prog_easu.log(),
                    )
                self._prog_easu.removeAllShaders()
                self._prog_easu = None
                return False

        if self._prog_rcas is None or not self._prog_rcas.isLinked():
            self._prog_rcas = QOpenGLShaderProgram()
            self._prog_rcas.bindAttributeLocation("a_pos_uv", 0)
            ok = self._prog_rcas.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, _VS
            )
            ok = ok and self._prog_rcas.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, rcas_src
            )
            ok = ok and self._prog_rcas.link()
            if not ok:
                if self._prog_rcas is not None:
                    print(
                        "[FSR1] RCAS shader link failed:",
                        self._prog_rcas.log(),
                    )
                self._prog_rcas.removeAllShaders()
                self._prog_rcas = None
                return False
        return True

    def _ensure_vbo(self) -> bool:
        if QOpenGLBuffer is None:
            return False
        if self._vbo is not None:
            return True
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vbo.create()
        return self._vbo.isCreated()

    def _ensure_vao(self) -> bool:
        """OpenGL Core: drawing with VBO attribs requires a bound VAO (VAO 0 is invalid)."""
        if QOpenGLVertexArrayObject is None:
            return False
        if self._vao is not None and self._vao.isCreated():
            return True
        v = QOpenGLVertexArrayObject()
        v.create()
        if not v.isCreated():
            return False
        self._vao = v
        return True

    def _src_gl_destroy(self, ctx: QtGui.QOpenGLContext | None) -> None:
        tid = int(self._tex_src_gl_id)
        backend = getattr(self, "_tex_src_backend", "none")
        raw_ctypes = getattr(self, "_tex_src_raw_ctypes", False)
        self._tex_src_gl_id = 0
        self._tex_src_gl_w = 0
        self._tex_src_gl_h = 0
        self._tex_src_backend = "none"
        self._tex_src_raw_ctypes = False

        qt_tex = self._tex_src_qt
        self._tex_src_qt = None
        if qt_tex is not None:
            try:
                qt_tex.destroy()
            except Exception:
                pass

        if backend != "raw" or tid <= 0 or ctx is None:
            return
        if QtGui.QOpenGLContext.currentContext() is not ctx:
            return
        if raw_ctypes:
            _fsr_gl_c_delete_textures(ctx, tid)
        else:
            glf = _gl_texture_functions(ctx)
            _gl_delete_texture_name(glf, tid)

    def _src_gl_ensure_raw(self, ctx: QtGui.QOpenGLContext, w: int, h: int) -> bool:
        self._tex_src_raw_ctypes = False

        tid = _fsr_gl_c_gen_textures(ctx)
        if tid > 0 and _fsr_gl_c_proc_addr(ctx, b"glTexImage2D") != 0:
            ok_ct = False
            try:
                _fsr_gl_c_active_texture(ctx, _GL_TEXTURE0)
                _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, tid)
                _fsr_gl_c_tex_parameter_i(
                    ctx, _GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_LINEAR
                )
                _fsr_gl_c_tex_parameter_i(
                    ctx, _GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_LINEAR
                )
                _fsr_gl_c_tex_parameter_i(
                    ctx, _GL_TEXTURE_2D, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE
                )
                _fsr_gl_c_tex_parameter_i(
                    ctx, _GL_TEXTURE_2D, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE
                )
                _fsr_gl_c_pixel_store_i(ctx, _GL_UNPACK_ALIGNMENT, 4)
                if not _fsr_gl_c_tex_image_2d_rgba8_empty(ctx, _GL_TEXTURE_2D, w, h):
                    raise RuntimeError("glTexImage2D(ctypes) falló")
                ok_ct = True
            except Exception as e:
                if _fsr1_debug_env():
                    _fsr1_dbg(f"_src_gl_ensure_raw (ctypes): {e!r}")
                _fsr_gl_c_delete_textures(ctx, tid)
                tid = 0
            finally:
                _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
            if ok_ct and tid > 0:
                self._tex_src_gl_id = tid
                self._tex_src_raw_ctypes = True
                if _fsr1_debug_gl():
                    _fsr1_dbg(
                        f"_src_gl_ensure_raw: textura vídeo OK vía getProcAddress+ctypes id={tid}"
                    )
                return True

        glf = _gl_texture_functions(ctx)
        need = (
            "glGenTextures",
            "glBindTexture",
            "glTexParameteri",
            "glTexImage2D",
            "glPixelStorei",
        )
        if glf is None or not all(hasattr(glf, n) for n in need):
            if _fsr1_debug_gl():
                miss = [n for n in need if glf is None or not hasattr(glf, n)]
                _fsr1_dbg(
                    f"_src_gl_ensure_raw: sin {miss} "
                    f"glf={type(glf).__name__ if glf is not None else 'None'}"
                )
            return False
        tid2 = _gl_gen_texture_name(glf)
        if tid2 <= 0:
            if _fsr1_debug_gl():
                _fsr1_dbg("_src_gl_ensure_raw: glGenTextures (PySide) devolvió 0")
            return False
        try:
            _fsr_gl_active_texture_unit0(ctx)
            glf.glBindTexture(_GL_TEXTURE_2D, tid2)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_LINEAR)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_LINEAR)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE)
            glf.glPixelStorei(_GL_UNPACK_ALIGNMENT, 4)
            try:
                glf.glTexImage2D(
                    _GL_TEXTURE_2D,
                    0,
                    _GL_RGBA8,
                    w,
                    h,
                    0,
                    _GL_RGBA,
                    _GL_UNSIGNED_BYTE,
                    None,
                )
            except TypeError:
                glf.glTexImage2D(
                    _GL_TEXTURE_2D,
                    0,
                    _GL_RGBA8,
                    w,
                    h,
                    0,
                    _GL_RGBA,
                    _GL_UNSIGNED_BYTE,
                    ctypes.c_void_p(0),
                )
        except Exception as e:
            if _fsr1_debug_env():
                _fsr1_dbg(f"_src_gl_ensure_raw (PySide): excepción {e!r}")
            _gl_delete_texture_name(glf, tid2)
            return False
        finally:
            try:
                glf.glBindTexture(_GL_TEXTURE_2D, 0)
            except Exception:
                pass

        self._tex_src_gl_id = tid2
        return True

    def _src_gl_ensure_qt(self, w: int, h: int) -> bool:
        if QOpenGLTexture is None:
            return False
        t = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        t.setSize(w, h)
        t.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        t.setMipLevels(1)
        t.setAutoMipMapGenerationEnabled(False)
        t.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        t.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        t.setWrapMode(
            QOpenGLTexture.CoordinateDirection.DirectionS,
            QOpenGLTexture.WrapMode.ClampToEdge,
        )
        t.setWrapMode(
            QOpenGLTexture.CoordinateDirection.DirectionT,
            QOpenGLTexture.WrapMode.ClampToEdge,
        )
        self._tex_src_qt = t
        return True

    def _src_gl_ensure(self, ctx: QtGui.QOpenGLContext, w: int, h: int) -> bool:
        w = max(1, int(w))
        h = max(1, int(h))
        b = getattr(self, "_tex_src_backend", "none")
        if (
            b == "raw"
            and self._tex_src_gl_id > 0
            and self._tex_src_gl_w == w
            and self._tex_src_gl_h == h
        ):
            return True
        if b == "qt" and self._tex_src_qt is not None:
            if self._tex_src_gl_w == w and self._tex_src_gl_h == h:
                try:
                    if int(self._tex_src_qt.width()) == w and int(
                        self._tex_src_qt.height()
                    ) == h:
                        return True
                except RuntimeError:
                    pass

        self._src_gl_destroy(ctx)

        if self._src_gl_ensure_raw(ctx, w, h):
            self._tex_src_backend = "raw"
            self._tex_src_gl_w = w
            self._tex_src_gl_h = h
            return True

        if _fsr1_debug_env():
            _fsr1_dbg(
                "FSR1: textura vídeo GL directa no disponible; usando QOpenGLTexture (fallback)"
            )
        if self._src_gl_ensure_qt(w, h):
            self._tex_src_backend = "qt"
            self._tex_src_gl_w = w
            self._tex_src_gl_h = h
            self._tex_src_gl_id = 0
            return True
        return False

    def _src_upload_tex_qt(self, ctx: QtGui.QOpenGLContext, rgba: np.ndarray) -> None:
        t = self._tex_src_qt
        if t is None:
            return
        h, w = int(rgba.shape[0]), int(rgba.shape[1])
        f = ctx.functions()
        ps = getattr(f, "glPixelStorei", None) if f is not None else None
        if ps is not None:
            ps(_GL_UNPACK_ALIGNMENT, 4)
        try:
            t.bind(0)
            t.setData(
                0,
                0,
                0,
                w,
                h,
                1,
                QOpenGLTexture.PixelFormat.RGBA,
                QOpenGLTexture.PixelType.UInt8,
                rgba.tobytes(),
            )
            t.release()
        finally:
            if ps is not None:
                ps(_GL_UNPACK_ALIGNMENT, 4)

    def _src_gl_upload_rgba(self, ctx: QtGui.QOpenGLContext, rgba: np.ndarray) -> None:
        h, w = int(rgba.shape[0]), int(rgba.shape[1])
        if rgba.shape[2] != 4:
            raise ValueError("expected HxWx4 uint8 RGBA")
        if getattr(self, "_tex_src_backend", "none") == "qt":
            self._src_upload_tex_qt(ctx, rgba)
            return

        if getattr(self, "_tex_src_raw_ctypes", False):
            _fsr_gl_c_active_texture(ctx, _GL_TEXTURE0)
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, int(self._tex_src_gl_id))
            _fsr_gl_c_pixel_store_i(ctx, _GL_UNPACK_ALIGNMENT, 4)
            _fsr_gl_c_tex_sub_image_2d_rgba(
                ctx, _GL_TEXTURE_2D, w, h, rgba
            )
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
            return

        glf = _gl_texture_functions(ctx)
        if glf is None or not hasattr(glf, "glTexSubImage2D"):
            return
        arr = np.ascontiguousarray(rgba, dtype=np.uint8)
        if not arr.flags.writeable:
            arr = arr.copy()
        buf = (ctypes.c_uint8 * arr.size).from_buffer(arr)
        ptr = ctypes.cast(buf, ctypes.c_void_p)
        _fsr_gl_active_texture_unit0(ctx)
        try:
            glf.glBindTexture(_GL_TEXTURE_2D, int(self._tex_src_gl_id))
            glf.glPixelStorei(_GL_UNPACK_ALIGNMENT, 4)
            try:
                glf.glTexSubImage2D(
                    _GL_TEXTURE_2D, 0, 0, 0, w, h, _GL_RGBA, _GL_UNSIGNED_BYTE, ptr
                )
            except TypeError:
                from PySide6.QtCore import QByteArray

                glf.glTexSubImage2D(
                    _GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    w,
                    h,
                    _GL_RGBA,
                    _GL_UNSIGNED_BYTE,
                    QByteArray(arr.tobytes()),
                )
        finally:
            try:
                glf.glBindTexture(_GL_TEXTURE_2D, 0)
            except Exception:
                pass

    def _src_gl_bind_unit0(self, ctx: QtGui.QOpenGLContext) -> None:
        if getattr(self, "_tex_src_backend", "none") == "qt":
            _fsr_gl_active_texture_unit0(ctx)
            if self._tex_src_qt is not None:
                self._tex_src_qt.bind(0)
            return
        if getattr(self, "_tex_src_raw_ctypes", False):
            _fsr_gl_c_active_texture(ctx, _GL_TEXTURE0)
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, int(self._tex_src_gl_id))
            return
        glf = _gl_texture_functions(ctx)
        if glf is None:
            return
        _fsr_gl_active_texture_unit0(ctx)
        glf.glBindTexture(_GL_TEXTURE_2D, int(self._tex_src_gl_id))

    def _src_gl_unbind_unit0(self, ctx: QtGui.QOpenGLContext) -> None:
        if getattr(self, "_tex_src_backend", "none") == "qt":
            if self._tex_src_qt is not None:
                self._tex_src_qt.release()
            return
        if getattr(self, "_tex_src_raw_ctypes", False):
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
            return
        glf = _gl_texture_functions(ctx)
        if glf is None:
            return
        glf.glBindTexture(_GL_TEXTURE_2D, 0)

    def _configure_fbo_texture_linear(
        self, ctx: QtGui.QOpenGLContext, fbo: Any
    ) -> bool:
        tid = int(fbo.texture())
        if tid <= 0:
            return False
        f = ctx.functions()
        f.glBindTexture(_GL_TEXTURE_2D, tid)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_LINEAR)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_LINEAR)
        f.glBindTexture(_GL_TEXTURE_2D, 0)
        return True

    def _ensure_fsr_pass_fbos(self, ctx: QtGui.QOpenGLContext, rw: int, rh: int) -> bool:
        """EASU+RCAS FBOs sized to the item's on-screen pixel rect (not full QGraphicsView)."""
        if QOpenGLFramebufferObject is None or QOpenGLFramebufferObjectFormat is None:
            return False
        rw = max(1, rw)
        rh = max(1, rh)
        if (
            self._fbo is not None
            and self._fbo_rcas is not None
            and self._fbo_w == rw
            and self._fbo_h == rh
        ):
            return True
        self._invalidate_fbo()
        fmt = QOpenGLFramebufferObjectFormat()
        try:
            fmt.setAttachment(QOpenGLFramebufferObject.Attachment.NoAttachment)
        except Exception:
            pass
        fmt.setSamples(0)
        try:
            self._fbo = QOpenGLFramebufferObject(rw, rh, fmt)
            self._fbo_rcas = QOpenGLFramebufferObject(rw, rh, fmt)
            self._fbo_w = rw
            self._fbo_h = rh
        except Exception:
            self._invalidate_fbo()
            return False
        if not self._configure_fbo_texture_linear(ctx, self._fbo):
            self._invalidate_fbo()
            return False
        if not self._configure_fbo_texture_linear(ctx, self._fbo_rcas):
            self._invalidate_fbo()
            return False
        return True

    def _item_device_pixel_size(self, painter: QtGui.QPainter) -> tuple[int, int]:
        """Raster size of the item in paint-device pixels (letterboxed rect in the viewport)."""
        rect = self.boundingRect()
        dt = painter.deviceTransform()
        xs: list[float] = []
        ys: list[float] = []
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            p = dt.map(QtCore.QPointF(cx, cy))
            xs.append(float(p.x()))
            ys.append(float(p.y()))
        rw = max(1, int(math.ceil(max(xs) - min(xs))))
        rh = max(1, int(math.ceil(max(ys) - min(ys))))
        return rw, rh

    def _ensure_vbo_fs(self) -> bool:
        if QOpenGLBuffer is None:
            return False
        if self._vbo_fs is not None and self._vbo_fs.isCreated():
            return True
        self._vbo_fs = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vbo_fs.create()
        if not self._vbo_fs.isCreated():
            return False
        self._vbo_fs.bind()
        self._vbo_fs.allocate(
            _FS_FULLSCREEN_NDC_UV.tobytes(), _FS_FULLSCREEN_NDC_UV.nbytes
        )
        self._vbo_fs.release()
        return True

    def _build_vertices(
        self, painter: QtGui.QPainter, ndc_w: int, ndc_h: int
    ) -> np.ndarray:
        """NDC from item corners; denominators match QGraphicsView→viewport logical size."""
        rect = self.boundingRect()
        dt = painter.deviceTransform()
        dw = max(1, int(ndc_w))
        dh = max(1, int(ndc_h))
        ndc_uv = []
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            p = dt.map(QtCore.QPointF(cx, cy))
            x_ndc = 2.0 * float(p.x()) / float(dw) - 1.0
            y_ndc = 1.0 - 2.0 * float(p.y()) / float(dh)
            ndc_uv.append((x_ndc, y_ndc))
        tl, tr, bl, br = ndc_uv
        arr = np.array(
            [
                *tl,
                0.0,
                0.0,
                *tr,
                1.0,
                0.0,
                *bl,
                0.0,
                1.0,
                *tr,
                1.0,
                0.0,
                *br,
                1.0,
                1.0,
                *bl,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        # deviceTransform + float rounding can yield |NDC| > 1; stay inside clip volume.
        arr[0::4] = np.clip(arr[0::4], -1.0, 1.0)
        arr[1::4] = np.clip(arr[1::4], -1.0, 1.0)
        return arr

    def _read_draw_framebuffer_id(
        self,
        f: Any,
        ctx: QtGui.QOpenGLContext | None,
        gl_widget: QtWidgets.QWidget | None,
    ) -> int | None:
        bid = _gl_query_framebuffer_binding(f, ctx, gl_widget)
        if bid is None or bid <= 0:
            return None
        return bid

    def _bind_draw_framebuffer(
        self,
        f: Any,
        fbo_id: int,
        gl_widget: QtWidgets.QWidget,
        ctx: QtGui.QOpenGLContext | None,
    ) -> None:
        _gl_bind_framebuffer(f, ctx, gl_widget, int(fbo_id))

    def _bind_qopengl_widget_backbuffer(
        self,
        gl_widget: QtWidgets.QWidget,
        f: Any,
        ctx: QtGui.QOpenGLContext | None,
    ) -> None:
        """Draw to the widget's real target; QOpenGLWidget is NOT framebuffer 0."""
        dfo = getattr(gl_widget, "defaultFramebufferObject", None)
        if dfo is None:
            return
        try:
            fid = int(dfo())
        except Exception:
            return
        if fid <= 0:
            return
        _gl_bind_framebuffer(f, ctx, gl_widget, fid)

    def _restore_qt_draw_framebuffer(
        self,
        f: Any,
        gl_widget: QtWidgets.QWidget,
        prev_fbo: int | None,
        ctx: QtGui.QOpenGLContext | None,
    ) -> None:
        """After rendering to our own FBO, bind back what Qt had (not always == defaultFramebufferObject)."""
        if prev_fbo is not None and int(prev_fbo) > 0:
            self._bind_draw_framebuffer(f, int(prev_fbo), gl_widget, ctx)
            return
        self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)

    def _draw_quad_like_blend(self, program: QOpenGLShaderProgram, f: Any) -> bool:
        """Attribs + draw; VAO bound for OpenGL Core (preview uses CoreProfile)."""
        loc = int(program.attributeLocation("a_pos_uv"))
        if loc < 0:
            loc = int(program.attributeLocation(b"a_pos_uv"))
        if loc < 0:
            return False
        vao = self._vao
        use_vao = (
            QOpenGLVertexArrayObject is not None
            and vao is not None
            and vao.isCreated()
        )
        if use_vao:
            vao.bind()
        try:
            program.enableAttributeArray(loc)
            program.setAttributeBuffer(loc, _GL_FLOAT, 0, 4, 0)
            f.glDrawArrays(_GL_TRIANGLES, 0, 6)
            program.disableAttributeArray(loc)
            return True
        finally:
            if use_vao:
                vao.release()

    def _set_uniform_vec2(self, program: QOpenGLShaderProgram, name: str, x: float, y: float) -> None:
        # PySide6: setUniformValue rejects str uniform names for many overloads; use location or ascii bytes.
        loc = _qgl_uniform_location(program, name)
        if loc >= 0:
            program.setUniformValue(loc, float(x), float(y))
            return
        try:
            program.setUniformValue(name.encode("ascii"), float(x), float(y))
        except Exception:
            pass

    def _set_uniform_float(self, program: QOpenGLShaderProgram, name: str, value: float) -> None:
        # PySide6 has no setUniformValue(uniformName: str|bytes, float); only (location: int, float).
        loc = _qgl_uniform_location(program, name)
        if loc < 0:
            loc = _qgl_uniform_location_gl(program, name)
        if loc >= 0:
            program.setUniformValue(loc, float(value))

    def _set_uniform_int(self, program: QOpenGLShaderProgram, name: str, value: int) -> None:
        loc = _qgl_uniform_location(program, name)
        if loc < 0:
            loc = _qgl_uniform_location_gl(program, name)
        if loc >= 0:
            program.setUniformValue(loc, int(value))


    def _viewport_corner_flat(self, gv: QtWidgets.QGraphicsView) -> tuple[float, ...]:
        rect = self.boundingRect()
        out: list[float] = []
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            vp = gv.mapFromScene(self.mapToScene(QtCore.QPointF(cx, cy)))
            out.extend((float(vp.x()), float(vp.y())))
        return tuple(out)

    def _item_device_pixel_size_from_view(
        self, gv: QtWidgets.QGraphicsView
    ) -> tuple[int, int]:
        rect = self.boundingRect()
        xs: list[float] = []
        ys: list[float] = []
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            vp = gv.mapFromScene(self.mapToScene(QtCore.QPointF(cx, cy)))
            xs.append(float(vp.x()))
            ys.append(float(vp.y()))
        rw = max(1, int(math.ceil(max(xs) - min(xs))))
        rh = max(1, int(math.ceil(max(ys) - min(ys))))
        return rw, rh

    def _build_vertices_from_view(
        self, gv: QtWidgets.QGraphicsView, ndc_w: int, ndc_h: int
    ) -> np.ndarray:
        rect = self.boundingRect()
        dw = max(1, int(ndc_w))
        dh = max(1, int(ndc_h))
        ndc_uv = []
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            vp = gv.mapFromScene(self.mapToScene(QtCore.QPointF(cx, cy)))
            px = float(vp.x())
            py = float(vp.y())
            x_ndc = 2.0 * px / float(dw) - 1.0
            y_ndc = 1.0 - 2.0 * py / float(dh)
            ndc_uv.append((x_ndc, y_ndc))
        tl, tr, bl, br = ndc_uv
        arr = np.array(
            [
                *tl,
                0.0,
                0.0,
                *tr,
                1.0,
                0.0,
                *bl,
                0.0,
                1.0,
                *tr,
                1.0,
                0.0,
                *br,
                1.0,
                1.0,
                *bl,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        arr[0::4] = np.clip(arr[0::4], -1.0, 1.0)
        arr[1::4] = np.clip(arr[1::4], -1.0, 1.0)
        return arr

    def render_gl_in_viewport(
        self,
        gl_widget: QtWidgets.QWidget,
        gv: QtWidgets.QGraphicsView,
    ) -> None:
        """Tras pintar la escena en QOpenGLWidget.paintEvent (sin beginNativePainting en el item)."""
        if self._frame_bgr is None or QOpenGLShaderProgram is None:
            return
        if self._gl_failed:
            if _fsr1_debug_env() and not getattr(
                self, "_fsr1_debug_logged_stuck", False
            ):
                self._fsr1_debug_logged_stuck = True
                _fsr1_dbg(
                    "render: skipped (_gl_failed=True); turn FSR off/on or restart to retry"
                )
            return

        if _fsr1_debug_level() >= 2:
            n = getattr(self, "_fsr1_trace_paint_n", 0) + 1
            self._fsr1_trace_paint_n = n
            if n <= 15 or n % 120 == 0:
                _fsr1_dbg(
                    f"render #{n} present_seq={self._present_seq} "
                    f"frame={getattr(self._frame_bgr, 'shape', None)}"
                )

        try:
            gl_widget.makeCurrent()
        except Exception as e:
            self._gl_failed = True
            _fsr1_dbg(f"makeCurrent failed: {e!r}")
            return

        ctx = gl_widget.context() or QtGui.QOpenGLContext.currentContext()
        if ctx is None:
            _fsr1_dbg("render_gl_in_viewport: no QOpenGLContext")
            return

        schedule_khr_debug_install_for_context(ctx)
        log_gl_driver_info_once(ctx)

        try:
            isolate = _fsr1_isolate_mode()
            if isolate in ("blit_src", "easu_only"):
                if not self._fsr1_isolate_logged:
                    self._fsr1_isolate_logged = True
                    _fsr1_dbg(
                        f"ISOLATE={isolate}: modo prueba (quita VISIOMASTER_FSR1_ISOLATE para pipeline completo)"
                    )
            elif isolate:
                if not self._fsr1_isolate_bad_logged:
                    self._fsr1_isolate_bad_logged = True
                    _fsr1_dbg(
                        f"VISIOMASTER_FSR1_ISOLATE desconocido={isolate!r}; use blit_src o easu_only"
                    )

            direct_blit = (not self._use_pipeline_shaders) or isolate == "blit_src"
            easu_only = bool(self._use_pipeline_shaders and isolate == "easu_only")

            _fsr1_gl_log_fb(ctx, gl_widget, "viewport overlay (post-scene)")
            _fsr1_gl_drain_errors(ctx, "incoming")

            self._apply_fsr_program_version()
            if direct_blit:
                if not self._ensure_blit_program():
                    self._gl_failed = True
                    _fsr1_dbg("blit shader build or link failed (see log above if printed)")
                    return
            else:
                if not self._ensure_fsr_programs():
                    self._gl_failed = True
                    _fsr1_dbg("EASU/RCAS shader build or link failed (see log above if printed)")
                    return
                if not self._ensure_blit_program():
                    self._gl_failed = True
                    _fsr1_dbg("blit shader build or link failed (FSR composite)")
                    return
            self._fsr1_gl_log_uniforms_once()
            if not self._ensure_vbo():
                self._gl_failed = True
                _fsr1_dbg("VBO create failed")
                return
            if not self._ensure_vao():
                self._gl_failed = True
                _fsr1_dbg(
                    "VAO create failed (Core profile requiere VAO para glDrawArrays con VBO)"
                )
                return

            h, w = self._frame_bgr.shape[:2]
            if not self._src_gl_ensure(ctx, w, h):
                self._gl_failed = True
                _fsr1_dbg("source texture allocation failed (glGenTextures/glTexImage2D)")
                return

            if self._present_seq != self._upload_applied_seq:
                rgb = _numpy_bgr_to_rgb_contiguous(self._frame_bgr)
                rgba = _numpy_rgb_to_rgba_contiguous(rgb)
                self._src_gl_upload_rgba(ctx, rgba)
                self._upload_applied_seq = self._present_seq

            try:
                gl_widget.makeCurrent()
            except Exception:
                pass

            vw = max(1, gl_widget.width())
            vh = max(1, gl_widget.height())
            vp_w, vp_h = _gl_viewport_phys_size(gl_widget)
            vbo_key = (
                vw,
                vh,
                self._lay_w,
                self._lay_h,
                0 if direct_blit else (2 if easu_only else 1),
                self._viewport_corner_flat(gv),
            )
            rebuild = vbo_key != self._vbo_cache_key or self._cached_verts is None
            if rebuild:
                self._cached_verts = self._build_vertices_from_view(gv, vw, vh)
                self._vbo_cache_key = vbo_key
            verts = self._cached_verts

            assert self._vbo is not None
            self._vbo.bind()
            if rebuild:
                self._vbo.allocate(verts.tobytes(), verts.nbytes)

            f = ctx.functions()
            f.glViewport(0, 0, vp_w, vp_h)
            f.glDisable(_GL_DEPTH_TEST)
            f.glDisable(_GL_BLEND)
            try:
                f.glDisable(_GL_SCISSOR_TEST)
            except Exception:
                pass
            try:
                f.glDisable(_GL_CULL_FACE)
            except Exception:
                pass

            rw, rh = self._item_device_pixel_size_from_view(gv)
            self._preview_overlay_src_wh = (int(w), int(h))
            self._preview_overlay_tgt_wh = (int(rw), int(rh))

            self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)
            _fsr1_gl_log_fb(ctx, gl_widget, "after bind widget draw target")

            if direct_blit:
                assert self._prog_blit is not None
                _fsr_gl_active_texture_unit0(ctx)
                self._prog_blit.bind()
                self._src_gl_bind_unit0(ctx)
                self._set_uniform_int(self._prog_blit, "u_src", 0)
                self._vbo.bind()
                if not self._draw_quad_like_blend(self._prog_blit, f):
                    self._gl_failed = True
                    _fsr1_dbg("blit draw failed (attribute a_pos_uv / GL state)")
                    self._vbo.release()
                    return
                self._src_gl_unbind_unit0(ctx)
                self._prog_blit.release()
                _fsr1_gl_drain_errors(ctx, "after direct blit")
                _fsr1_gl_log_fb(ctx, gl_widget, "after direct blit")
            else:
                if (
                    not self._ensure_fsr_pass_fbos(ctx, rw, rh)
                    or self._fbo is None
                    or self._fbo_rcas is None
                ):
                    self._gl_failed = True
                    _fsr1_dbg(f"FSR pass FBO create failed ({rw}x{rh})")
                    self._vbo.release()
                    return

                if not self._ensure_vbo_fs():
                    self._gl_failed = True
                    _fsr1_dbg("fullscreen VBO create failed")
                    self._vbo.release()
                    return

                assert self._prog_easu is not None
                assert self._prog_rcas is not None
                raw_bind_pre = _gl_query_framebuffer_binding(f, ctx, gl_widget)
                prev_qt_fbo = self._read_draw_framebuffer_id(f, ctx, gl_widget)
                if _fsr1_debug_gl():
                    tid0 = int(self._fbo.texture())
                    tid1 = int(self._fbo_rcas.texture())
                    _fsr1_dbg(
                        f"pre-pass FBO tex ids easu_out={tid0} rcas_out={tid1} passSize={rw}x{rh} "
                        f"raw_BINDING={raw_bind_pre} restore_target_filtered={prev_qt_fbo!r}"
                    )

                self._fbo.bind()
                f.glViewport(0, 0, rw, rh)
                _fsr1_gl_log_fb(ctx, gl_widget, "bound EASU FBO")
                self._vbo_fs.bind()
                self._prog_easu.bind()
                _fsr_gl_active_texture_unit0(ctx)
                self._src_gl_bind_unit0(ctx)
                self._set_uniform_int(self._prog_easu, "u_src", 0)
                self._set_uniform_vec2(self._prog_easu, "u_inSize", float(w), float(h))
                self._set_uniform_vec2(self._prog_easu, "u_outSize", float(rw), float(rh))
                if not self._draw_quad_like_blend(self._prog_easu, f):
                    self._gl_failed = True
                    _fsr1_dbg("EASU draw failed")
                    self._vbo_fs.release()
                    self._vbo.release()
                    return
                _fsr1_gl_drain_errors(ctx, "after EASU")
                # Coherencia escritura FBO EASU → muestreo en RCAS (evita GL_INVALID_OPERATION en NVIDIA).
                gl_texture_barrier_after_image_write(ctx)
                # No llamar _fbo.release()/bindDefault(): en QOpenGLWidget eso suele enlazar FBO 0
                # (incorrecto). El siguiente bind() del FBO destino cambia el draw target.
                self._src_gl_unbind_unit0(ctx)
                self._prog_easu.release()

                if easu_only:
                    if not self._ensure_blit_program():
                        self._gl_failed = True
                        _fsr1_dbg("blit program missing (easu_only)")
                        self._vbo_fs.release()
                        self._vbo.release()
                        return
                    self._restore_qt_draw_framebuffer(f, gl_widget, prev_qt_fbo, ctx)
                    _fsr1_gl_log_fb(ctx, gl_widget, "after restore (easu_only)")
                    f.glViewport(0, 0, vp_w, vp_h)
                    self._vbo.bind()
                    self._prog_blit.bind()
                    tid_easu_out = int(self._fbo.texture())
                    _fsr_gl_active_texture_unit0(ctx)
                    f.glBindTexture(_GL_TEXTURE_2D, tid_easu_out)
                    self._set_uniform_int(self._prog_blit, "u_src", 0)
                    if not self._draw_quad_like_blend(self._prog_blit, f):
                        self._gl_failed = True
                        _fsr1_dbg("easu_only composite blit failed")
                        f.glBindTexture(_GL_TEXTURE_2D, 0)
                        self._prog_blit.release()
                        self._vbo_fs.release()
                        self._vbo.release()
                        return
                    f.glBindTexture(_GL_TEXTURE_2D, 0)
                    self._prog_blit.release()
                    self._vbo_fs.release()
                    _fsr1_gl_drain_errors(ctx, "after easu_only composite")
                    _fsr1_gl_log_fb(ctx, gl_widget, "after easu_only composite")
                else:
                    self._fbo_rcas.bind()
                    f.glViewport(0, 0, rw, rh)
                    _fsr1_gl_log_fb(ctx, gl_widget, "bound RCAS FBO")
                    self._prog_rcas.bind()
                    tid_easu = int(self._fbo.texture())
                    _fsr_gl_active_texture_unit0(ctx)
                    f.glBindTexture(_GL_TEXTURE_2D, tid_easu)
                    self._set_uniform_int(self._prog_rcas, "u_tex", 0)
                    self._set_uniform_vec2(self._prog_rcas, "u_outSize", float(rw), float(rh))
                    self._set_uniform_float(
                        self._prog_rcas, "u_sharpness", float(self._sharpness)
                    )
                    if not self._draw_quad_like_blend(self._prog_rcas, f):
                        self._gl_failed = True
                        _fsr1_dbg("RCAS draw failed")
                        f.glBindTexture(_GL_TEXTURE_2D, 0)
                        self._prog_rcas.release()
                        self._vbo_fs.release()
                        self._vbo.release()
                        return
                    _fsr1_gl_drain_errors(ctx, "after RCAS")
                    gl_texture_barrier_after_image_write(ctx)
                    f.glBindTexture(_GL_TEXTURE_2D, 0)
                    self._prog_rcas.release()

                    if not self._ensure_blit_program():
                        self._gl_failed = True
                        _fsr1_dbg("blit program missing for FSR composite")
                        self._vbo_fs.release()
                        self._vbo.release()
                        return

                    self._restore_qt_draw_framebuffer(f, gl_widget, prev_qt_fbo, ctx)
                    _fsr1_gl_log_fb(ctx, gl_widget, "after restore (full pipeline)")
                    f.glViewport(0, 0, vp_w, vp_h)
                    self._vbo.bind()
                    self._prog_blit.bind()
                    tid_rcas = int(self._fbo_rcas.texture())
                    _fsr_gl_active_texture_unit0(ctx)
                    f.glBindTexture(_GL_TEXTURE_2D, tid_rcas)
                    self._set_uniform_int(self._prog_blit, "u_src", 0)
                    if not self._draw_quad_like_blend(self._prog_blit, f):
                        self._gl_failed = True
                        _fsr1_dbg("FSR composite blit failed")
                        f.glBindTexture(_GL_TEXTURE_2D, 0)
                        self._prog_blit.release()
                        self._vbo_fs.release()
                        self._vbo.release()
                        return
                    f.glBindTexture(_GL_TEXTURE_2D, 0)
                    self._prog_blit.release()
                    self._vbo_fs.release()
                    _fsr1_gl_drain_errors(ctx, "after full composite blit")
                    _fsr1_gl_log_fb(ctx, gl_widget, "after full composite blit")

            iso_now = _fsr1_isolate_mode()
            if iso_now == "blit_src":
                path_key = "blit_isolate"
            elif not self._use_pipeline_shaders:
                path_key = "blit_ui"
            elif easu_only:
                path_key = "easu_only"
            else:
                path_key = "full_easu_rcas"
            if path_key != self._fsr1_pipeline_path_key:
                self._fsr1_pipeline_path_key = path_key
                if path_key == "blit_isolate":
                    modo = (
                        "Pipeline efectivo = solo blit de textura (sin EASU ni RCAS) porque "
                        "VISIOMASTER_FSR1_ISOLATE=blit_src. Eso anula el toggle «FSR1: EASU+RCAS shaders» "
                        "aunque esté activado en ajustes. Quita la variable (y reinicia desde esa terminal) "
                        "para usar el shader FSR real."
                    )
                elif path_key == "blit_ui":
                    modo = (
                        "Pipeline efectivo = solo blit: «FSR1: EASU+RCAS shaders» está desactivado en ajustes. "
                        "Actívalo para EASU + RCAS."
                    )
                elif path_key == "easu_only":
                    modo = (
                        "Pipeline efectivo = EASU solamente (sin RCAS) por "
                        "VISIOMASTER_FSR1_ISOLATE=easu_only. Quita ISOLATE para RCAS."
                    )
                else:
                    modo = (
                        "Pipeline efectivo = EASU + RCAS (FSR1). La salida escala al tamaño del preview; "
                        "si casi no ves cambio, amplía la ventana o sube «RCAS sharpness»."
                    )
                print(f"[FSR1] {modo} Más detalle: VISIOMASTER_DEBUG_FSR1=1.", flush=True)

            if _fsr1_debug_env() and not getattr(self, "_fsr1_debug_logged_ok", False):
                self._fsr1_debug_logged_ok = True
                xs = verts[0::4]
                ys = verts[1::4]
                dev_cls = type(gl_widget).__name__
                try:
                    dpr = float(gl_widget.devicePixelRatioF())
                    dfbo = int(gl_widget.defaultFramebufferObject())
                except Exception:
                    dpr = -1.0
                    dfbo = -1
                try:
                    fmean = float(np.mean(self._frame_bgr))
                    fmax = float(np.max(self._frame_bgr))
                except Exception:
                    fmean = -1.0
                    fmax = -1.0
                fsr_pass = ""
                if not direct_blit:
                    rpw, rph = self._item_device_pixel_size_from_view(gv)
                    fsr_pass = f" fsrPassFbo={rpw}x{rph}"
                path = (
                    "direct_blit"
                    if direct_blit
                    else ("easu_only" if easu_only else "full_easu_rcas")
                )
                iso = _fsr1_isolate_mode() or "-"
                _fsr1_dbg(
                    f"first successful render (viewport overlay): path={path} isolate={iso} "
                    f"ui_pipeline_shaders={self._use_pipeline_shaders} "
                    f"glWidget={dev_cls} logical={vw}x{vh} glViewport={vp_w}x{vp_h} "
                    f"tex={w}x{h} layout={self._lay_w}x{self._lay_h} "
                    f"widgetSize={gl_widget.width()}x{gl_widget.height()} dpr={dpr} defaultFBO={dfbo} "
                    f"present_seq={self._present_seq} frame_mean={fmean:.1f} frame_max={fmax:.0f} "
                    f"NDC_x=[{float(np.min(xs)):.4f},{float(np.max(xs)):.4f}] "
                    f"NDC_y=[{float(np.min(ys)):.4f},{float(np.max(ys)):.4f}]{fsr_pass}"
                )
            self._vbo.release()
        except Exception as e:
            self._gl_failed = True
            if _fsr1_debug_env():
                traceback.print_exc()
                _fsr1_dbg(f"render_gl_in_viewport exception: {e!r}")

    def paint(  # noqa: N802
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None,
    ) -> None:
        del option, painter, widget
        return
    def reset_gl_state(self) -> None:
        self._gl_failed = False
        self._fsr1_debug_logged_stuck = False
        self._fsr1_debug_logged_ok = False
        self._fsr1_trace_paint_n = 0
        self._upload_applied_seq = -1
        self._vbo_cache_key = None
        self._cached_verts = None
        self._fsr1_gl_extra_logged = False
        self._fsr1_isolate_logged = False
        self._fsr1_isolate_bad_logged = False
        self._fsr1_pipeline_path_key = None
        self._preview_overlay_src_wh = None
        self._preview_overlay_tgt_wh = None
        self._destroy_gl_objects()

    def _fsr1_gl_log_uniforms_once(self) -> None:
        if not _fsr1_debug_gl() or self._fsr1_gl_extra_logged:
            return
        self._fsr1_gl_extra_logged = True

        def line(prog: QOpenGLShaderProgram | None, label: str, names: tuple[str, ...]) -> None:
            if prog is None:
                return
            pid = int(prog.programId())
            chunks: list[str] = []
            for nm in names:
                ql = _qgl_uniform_location(prog, nm)
                gl = _qgl_uniform_location_gl(prog, nm)
                chunks.append(f"{nm}:qt{ql}/gl{gl}")
            _fsr1_dbg(f"uniforms [{label}] programId={pid} " + " ".join(chunks))

        line(self._prog_blit, "blit", ("u_src",))
        line(self._prog_easu, "easu", ("u_src", "u_inSize", "u_outSize"))
        line(self._prog_rcas, "rcas", ("u_tex", "u_outSize", "u_sharpness"))
        al_blit = -99
        if self._prog_blit is not None:
            al_blit = int(self._prog_blit.attributeLocation("a_pos_uv"))
            if al_blit < 0:
                al_blit = int(self._prog_blit.attributeLocation(b"a_pos_uv"))
        _fsr1_dbg(f"attrib a_pos_uv (blit) location={al_blit}")
