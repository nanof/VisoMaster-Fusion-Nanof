"""
NVIDIA Image Scaling (NVScaler) for video preview — OpenGL 4.3+ compute + blit.

Shader: app/ui/widgets/shaders/nis/nis_scaler.comp (logic from NVIDIAGameWorks/NVIDIAImageScaling, MIT).
Preview only; recording/export unchanged. Requires upscale (on-screen rect larger than source):
if the preview downscales the frame (dst smaller than texture), falls back to blit.
Upscale uses NVScaler for large ratios (not limited to 2×); see VISIOMASTER_NIS_MIN_K_SCALE.

Mutex with FSR1 preview: only one spatial upscale path active (see control handlers).

OpenGL: VISIOMASTER_GL_DEBUG=1 activa KHR_debug; NOTIFICATION con tope (cap) salvo
VISIOMASTER_GL_DEBUG_NOTIFICATIONS=1. Vendor/renderer: mismo flag o VISIOMASTER_DEBUG_NIS=1.
Ruta NIS: VISIOMASTER_DEBUG_NIS=1 (también drena glGetError y trazas «trace #N» paso a paso).
  Si el log se corta al cerrar: PYTHONUNBUFFERED=1 (o python -u) y lanzar sin launcher que oculte stderr.
Upscale >2×: varios pasos NVScaler (ping-pong FBO); un solo paso con k_scale<0.5 suele dar negro.
Tras compute se desenlaza la image unit 0 antes del blit (misma textura image+sampler → negro en NVIDIA).
Opcional: VISIOMASTER_NIS_NO_TEXTURE_GATHER=1 sustituye NIS_TEXTURE_GATHER en nis_scaler.comp (sin romper #version).
"""

from __future__ import annotations

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

from app.ui.widgets.nis_coef_tables import COEF_SCALE, COEF_USM
from app.ui.widgets.nis_config_cpu import (
    nis_max_compute_output_pixels,
    nis_upscale_chain,
    pack_nis_scaler_ubo,
)

from app.ui.widgets.video_preview_fsr_gl_item import (
    _BLIT_FS,
    _VS,
    _fsr1_gl_drain_errors,
    gl_texture_barrier_after_image_write,
    log_gl_driver_info_once,
    schedule_khr_debug_install_for_context,
    _gl_delete_texture_name,
    _gl_gen_texture_name,
    _fsr_gl_active_texture_unit0,
    _fsr_gl_c_active_texture,
    _fsr_gl_c_bind_texture,
    _fsr_gl_c_delete_textures,
    _fsr_gl_c_gen_textures,
    _fsr_gl_c_proc_addr,
    _fsr_gl_c_pixel_store_i,
    _fsr_gl_c_tex_image_2d_rgba8_empty,
    _fsr_gl_c_tex_image_2d_rgba32f,
    _fsr_gl_c_tex_parameter_i,
    _fsr_gl_c_tex_sub_image_2d_rgba,
    _gl_base_functions,
    _gl_bind_framebuffer,
    _gl_extra_for_context,
    _gl_query_framebuffer_binding,
    _gl_texture_functions,
    _gl_viewport_phys_size,
    _numpy_bgr_to_rgb_contiguous,
    _numpy_rgb_to_rgba_contiguous,
    _qgl_uniform_location,
    _qgl_uniform_location_gl,
    _GL_CLAMP_TO_EDGE,
    _GL_CULL_FACE,
    _GL_FLOAT,
    _GL_FRAMEBUFFER,
    _GL_LINEAR,
    _GL_RGBA,
    _GL_RGBA8,
    _GL_TEXTURE0,
    _GL_TEXTURE_2D,
    _GL_TEXTURE_MAG_FILTER,
    _GL_TEXTURE_MIN_FILTER,
    _GL_TEXTURE_WRAP_S,
    _GL_TEXTURE_WRAP_T,
    _GL_TRIANGLES,
    _GL_UNSIGNED_BYTE,
)

_GL_NEAREST = 0x2600
_GL_RGBA32F = 0x8814
_GL_TEXTURE1 = 0x84C1
_GL_TEXTURE2 = 0x84C2
_GL_UNIFORM_BUFFER = 0x8A11
_GL_WRITE_ONLY = 0x88B9
_GL_READ_ONLY = 0x88B8
_NIS_BLOCK_W = 32
_NIS_BLOCK_H = 24
_GL_SHADER_IMAGE_ACCESS_BARRIER_BIT = 0x00000020
_GL_TEXTURE_FETCH_BARRIER_BIT = 0x00000008

_SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


def _nis_debug() -> bool:
    v = os.environ.get("VISIOMASTER_DEBUG_NIS", "").strip().lower()
    return v in ("1", "true", "yes", "on", "all")


def _nis_dbg(msg: str) -> None:
    if _nis_debug():
        print(f"[NIS] {msg}", flush=True)


_NIS_TRACE_SEQ = [0]


def _nis_render_trace(msg: str) -> None:
    """Breadcrumb con flush agresivo (TDR/crash del driver a veces no dejan buffer estándar)."""
    if not _nis_debug():
        return
    _NIS_TRACE_SEQ[0] += 1
    print(f"[NIS] trace #{_NIS_TRACE_SEQ[0]}: {msg}", flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


def _nis_emit_shader_log(title: str, log: str | None) -> None:
    """Imprime el log del driver (compilación/enlace); si está vacío, indica cómo depurar."""
    body = (log or "").strip()
    print(f"[NIS] {title}", flush=True)
    if not body:
        print(
            "[NIS]   (log vacío: ejecuta la app desde PowerShell/CMD o activa "
            "VISIOMASTER_GL_DEBUG=1 + contexto debug para KHR_debug)",
            flush=True,
        )
        return
    for ln in body.splitlines():
        print(f"[NIS]   {ln}", flush=True)


def _nis_log_stderr_hint_once() -> None:
    """Si no hay consola (Windows .exe), los prints no se ven; avisar una vez."""
    if not _nis_debug():
        return
    if getattr(_nis_log_stderr_hint_once, "_done", False):
        return
    setattr(_nis_log_stderr_hint_once, "_done", True)
    print(
        "[NIS] Debug: salida va a stdout. Si no ves mensajes, lanza la app desde "
        "PowerShell/CMD tras: $env:VISIOMASTER_DEBUG_NIS='1'",
        flush=True,
    )


def _load_compute_source() -> str:
    p = _SHADERS_DIR / "nis" / "nis_scaler.comp"
    body = p.read_text(encoding="utf-8")
    raw = os.environ.get("VISIOMASTER_NIS_NO_TEXTURE_GATHER", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        # No anteponer #define: en GLSL #version debe ser la primera directiva.
        body = body.replace("#define NIS_TEXTURE_GATHER 1", "#define NIS_TEXTURE_GATHER 0", 1)
    return body


def _nis_gl_unbind_image_unit0(xf: Any) -> None:
    """Quitar image2D de la unidad 0 antes de muestrear la misma textura (evita negro en algunos drivers)."""
    bind_img = getattr(xf, "glBindImageTexture", None)
    if not callable(bind_img):
        return
    try:
        bind_img(0, 0, 0, False, 0, _GL_READ_ONLY, _GL_RGBA8)
    except Exception:
        pass


def _coef_rgba_planes() -> tuple[np.ndarray, np.ndarray]:
    """2×64 RGBA32F texels for coef_scaler / coef_usm (see NIS LoadFilterBanksSh)."""
    h, w = 64, 2
    s = np.zeros((h, w, 4), dtype=np.float32)
    u = np.zeros((h, w, 4), dtype=np.float32)
    for p in range(64):
        s[p, 0, :4] = COEF_SCALE[p, :4]
        s[p, 1, 0:2] = COEF_SCALE[p, 4:6]
        u[p, 0, :4] = COEF_USM[p, :4]
        u[p, 1, 0:2] = COEF_USM[p, 4:6]
    return s, u


class VideoPreviewNisGlItem(QtWidgets.QGraphicsObject):
    """NVScaler (compute) → FBO RGBA8 → blit to QOpenGLWidget backbuffer."""

    def __init__(self) -> None:
        super().__init__()
        self._frame_bgr: np.ndarray | None = None
        self._sharpness_01: float = 0.5
        self._use_compute: bool = True
        self._lay_w: int = 1
        self._lay_h: int = 1
        self._display_frame: int = -1
        self._present_seq: int = 0
        self._upload_applied_seq: int = -1
        self._prog_compute: QOpenGLShaderProgram | None = None
        self._prog_blit: QOpenGLShaderProgram | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._vao: Any = None
        self._ubo: QOpenGLBuffer | None = None
        self._tex_src_gl_id: int = 0
        self._tex_src_gl_w: int = 0
        self._tex_src_gl_h: int = 0
        self._tex_src_qt: Any = None
        self._tex_src_backend: str = "none"
        self._tex_src_raw_ctypes: bool = False
        self._tex_coef_s: int = 0
        self._tex_coef_u: int = 0
        self._coef_gl_ready: bool = False
        self._fbo: QOpenGLFramebufferObject | None = None
        self._fbo_w: int = 0
        self._fbo_h: int = 0
        self._fbo_aux: QOpenGLFramebufferObject | None = None
        self._fbo_aux_w: int = 0
        self._fbo_aux_h: int = 0
        self._gl_failed: bool = False
        self._vbo_cache_key: tuple[Any, ...] | None = None
        self._cached_verts: np.ndarray | None = None
        self._compute_ok_logged: bool = False
        self._preview_overlay_src_wh: tuple[int, int] | None = None
        self._preview_overlay_tgt_wh: tuple[int, int] | None = None
        self._nis_dbg_path_key: tuple[Any, ...] | None = None
        self._nis_dbg_no_ctx_logged: bool = False
        self.setZValue(1.0)
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802
        return QtCore.QRectF(0.0, 0.0, float(self._lay_w), float(self._lay_h))

    def set_frame_sharpness(
        self,
        texture_bgr: np.ndarray,
        sharpness_0_1: float,
        *,
        layout_hw: tuple[int, int],
        display_frame: int,
        use_compute_shaders: bool = True,
    ) -> None:
        use_c = bool(use_compute_shaders)
        if use_c != getattr(self, "_use_compute", True):
            self._vbo_cache_key = None
            self._cached_verts = None
        self._use_compute = use_c
        lh, lw = int(layout_hw[0]), int(layout_hw[1])
        if lw != self._lay_w or lh != self._lay_h:
            self.prepareGeometryChange()
            self._lay_w = lw
            self._lay_h = lh
            self._vbo_cache_key = None
            self._cached_verts = None
            self._invalidate_fbo()
        tw, th = int(texture_bgr.shape[1]), int(texture_bgr.shape[0])
        if tw != getattr(self, "_last_tw", -1) or th != getattr(self, "_last_th", -1):
            self._upload_applied_seq = -1
            self._last_tw = tw
            self._last_th = th
        self._frame_bgr = texture_bgr
        self._sharpness_01 = float(max(0.0, min(1.0, sharpness_0_1)))
        self._display_frame = int(display_frame)
        self._present_seq += 1
        self.update()

    def _invalidate_fbo(self) -> None:
        self._fbo = None
        self._fbo_w = 0
        self._fbo_h = 0
        self._fbo_aux = None
        self._fbo_aux_w = 0
        self._fbo_aux_h = 0

    def _destroy_gl_objects(self) -> None:
        self._invalidate_fbo()
        ctx = QtGui.QOpenGLContext.currentContext()
        try:
            self._src_gl_destroy(ctx)
        except Exception:
            self._tex_src_gl_id = 0
        for tid in (self._tex_coef_s, self._tex_coef_u):
            if tid > 0 and ctx is not None:
                try:
                    _fsr_gl_c_delete_textures(ctx, int(tid))
                except Exception:
                    pass
        self._tex_coef_s = 0
        self._tex_coef_u = 0
        self._coef_gl_ready = False
        if self._vbo is not None:
            self._vbo.destroy()
            self._vbo = None
        if self._vao is not None:
            try:
                self._vao.destroy()
            except Exception:
                pass
            self._vao = None
        if self._ubo is not None:
            self._ubo.destroy()
            self._ubo = None
        if self._prog_compute is not None:
            self._prog_compute.removeAllShaders()
        if self._prog_blit is not None:
            self._prog_blit.removeAllShaders()
        self._prog_compute = None
        self._prog_blit = None
        self._upload_applied_seq = -1
        self._vbo_cache_key = None
        self._cached_verts = None
        self._compute_ok_logged = False

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
            from app.ui.widgets.video_preview_fsr_gl_item import _gl_delete_texture_name

            _gl_delete_texture_name(glf, tid)

    def _src_gl_ensure_raw(self, ctx: QtGui.QOpenGLContext, w: int, h: int) -> bool:
        self._tex_src_raw_ctypes = False
        tid = _fsr_gl_c_gen_textures(ctx)
        if tid > 0 and _fsr_gl_c_proc_addr(ctx, b"glTexImage2D") != 0:
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
                _fsr_gl_c_pixel_store_i(ctx, 0x0CF5, 4)
                if not _fsr_gl_c_tex_image_2d_rgba8_empty(ctx, _GL_TEXTURE_2D, w, h):
                    raise RuntimeError("glTexImage2D")
                self._tex_src_gl_id = tid
                self._tex_src_raw_ctypes = True
                _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
                return True
            except Exception:
                _fsr_gl_c_delete_textures(ctx, tid)
        glf = _gl_texture_functions(ctx)
        tid2 = _gl_gen_texture_name(glf)
        if tid2 <= 0:
            return False
        try:
            _fsr_gl_active_texture_unit0(ctx)
            glf.glBindTexture(_GL_TEXTURE_2D, tid2)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_LINEAR)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_LINEAR)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE)
            glf.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE)
            glf.glPixelStorei(0x0CF5, 4)
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
                import ctypes as _ct

                glf.glTexImage2D(
                    _GL_TEXTURE_2D,
                    0,
                    _GL_RGBA8,
                    w,
                    h,
                    0,
                    _GL_RGBA,
                    _GL_UNSIGNED_BYTE,
                    _ct.c_void_p(0),
                )
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
        if self._src_gl_ensure_qt(w, h):
            self._tex_src_backend = "qt"
            self._tex_src_gl_w = w
            self._tex_src_gl_h = h
            self._tex_src_gl_id = 0
            return True
        return False

    def _src_gl_upload_rgba(self, ctx: QtGui.QOpenGLContext, rgba: np.ndarray) -> None:
        h, w = int(rgba.shape[0]), int(rgba.shape[1])
        if getattr(self, "_tex_src_backend", "none") == "qt":
            t = self._tex_src_qt
            if t is None:
                return
            f = ctx.functions()
            ps = getattr(f, "glPixelStorei", None) if f is not None else None
            if ps is not None:
                ps(0x0CF5, 4)
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
                    ps(0x0CF5, 4)
            return
        if getattr(self, "_tex_src_raw_ctypes", False):
            _fsr_gl_c_active_texture(ctx, _GL_TEXTURE0)
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, int(self._tex_src_gl_id))
            _fsr_gl_c_pixel_store_i(ctx, 0x0CF5, 4)
            _fsr_gl_c_tex_sub_image_2d_rgba(ctx, _GL_TEXTURE_2D, w, h, rgba)
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
            return
        glf = _gl_texture_functions(ctx)
        if glf is None or not hasattr(glf, "glTexSubImage2D"):
            return
        import ctypes

        arr = np.ascontiguousarray(rgba, dtype=np.uint8)
        if not arr.flags.writeable:
            arr = arr.copy()
        buf = (ctypes.c_uint8 * arr.size).from_buffer(arr)
        ptr = ctypes.cast(buf, ctypes.c_void_p)
        _fsr_gl_active_texture_unit0(ctx)
        try:
            glf.glBindTexture(_GL_TEXTURE_2D, int(self._tex_src_gl_id))
            glf.glPixelStorei(0x0CF5, 4)
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
        if glf is not None:
            glf.glBindTexture(_GL_TEXTURE_2D, 0)

    def _ensure_coef_textures(self, ctx: QtGui.QOpenGLContext) -> bool:
        if self._coef_gl_ready and self._tex_coef_s > 0 and self._tex_coef_u > 0:
            return True
        for tid in (self._tex_coef_s, self._tex_coef_u):
            if tid > 0:
                try:
                    _fsr_gl_c_delete_textures(ctx, int(tid))
                except Exception:
                    pass
        self._tex_coef_s = 0
        self._tex_coef_u = 0
        s_plane, u_plane = _coef_rgba_planes()
        h, w = s_plane.shape[0], s_plane.shape[1]
        glf = _gl_texture_functions(ctx)

        def _make_coef_tid(data: np.ndarray) -> int:
            # PySide6 a veces devuelve tipos raros desde glGenTextures; la textura vídeo
            # raw ya usa _fsr_gl_c_gen_textures — mismo fallback aquí (evita tid==0).
            tid = _gl_gen_texture_name(glf) if glf is not None else 0
            if tid <= 0:
                tid = _fsr_gl_c_gen_textures(ctx)
            if tid <= 0:
                return 0
            arr = np.ascontiguousarray(data, dtype=np.float32)
            if not arr.flags.writeable:
                arr = arr.copy()
            import ctypes

            buf = (ctypes.c_float * arr.size).from_buffer(arr)
            ptr_addr = int(ctypes.addressof(buf))
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, tid)
            _fsr_gl_c_tex_parameter_i(ctx, _GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_NEAREST)
            _fsr_gl_c_tex_parameter_i(ctx, _GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_NEAREST)
            _fsr_gl_c_tex_parameter_i(ctx, _GL_TEXTURE_2D, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE)
            _fsr_gl_c_tex_parameter_i(ctx, _GL_TEXTURE_2D, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE)
            ok = _fsr_gl_c_tex_image_2d_rgba32f(ctx, _GL_TEXTURE_2D, w, h, ptr_addr)
            _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
            if not ok:
                _fsr_gl_c_delete_textures(ctx, tid)
                return 0
            return tid

        ts = _make_coef_tid(s_plane)
        tu = _make_coef_tid(u_plane)
        if ts <= 0 or tu <= 0:
            if ts > 0:
                _gl_delete_texture_name(glf, ts)
            if tu > 0:
                _gl_delete_texture_name(glf, tu)
            return False
        self._tex_coef_s = ts
        self._tex_coef_u = tu
        self._coef_gl_ready = True
        return True

    def _configure_fbo_texture_linear(self, ctx: QtGui.QOpenGLContext, fbo: Any) -> bool:
        tid = int(fbo.texture())
        if tid <= 0:
            return False
        f = ctx.functions()
        f.glBindTexture(_GL_TEXTURE_2D, tid)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_LINEAR)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_LINEAR)
        f.glBindTexture(_GL_TEXTURE_2D, 0)
        return True

    def _ensure_nis_fbo_slot(
        self, ctx: QtGui.QOpenGLContext, *, primary: bool, tw: int, th: int
    ) -> bool:
        if QOpenGLFramebufferObject is None or QOpenGLFramebufferObjectFormat is None:
            return False
        tw = max(1, int(tw))
        th = max(1, int(th))
        if primary:
            if self._fbo is not None and self._fbo_w == tw and self._fbo_h == th:
                return True
            self._fbo = None
            self._fbo_w = 0
            self._fbo_h = 0
        else:
            if self._fbo_aux is not None and self._fbo_aux_w == tw and self._fbo_aux_h == th:
                return True
            self._fbo_aux = None
            self._fbo_aux_w = 0
            self._fbo_aux_h = 0
        fmt = QOpenGLFramebufferObjectFormat()
        try:
            fmt.setAttachment(QOpenGLFramebufferObject.Attachment.NoAttachment)
        except Exception:
            pass
        fmt.setSamples(0)
        try:
            fmt.setInternalTextureFormat(_GL_RGBA8)
        except Exception:
            pass
        try:
            fbo = QOpenGLFramebufferObject(tw, th, fmt)
        except Exception:
            return False
        if not self._configure_fbo_texture_linear(ctx, fbo):
            return False
        if primary:
            self._fbo = fbo
            self._fbo_w = tw
            self._fbo_h = th
        else:
            self._fbo_aux = fbo
            self._fbo_aux_w = tw
            self._fbo_aux_h = th
        return True

    def _ensure_nis_fbo(self, ctx: QtGui.QOpenGLContext, rw: int, rh: int) -> bool:
        return self._ensure_nis_fbo_slot(ctx, primary=True, tw=rw, th=rh)

    def _ensure_compute(self) -> bool:
        if QOpenGLShaderProgram is None or QOpenGLShader is None:
            return False
        if self._prog_compute is not None and self._prog_compute.isLinked():
            return True
        src = _load_compute_source()
        cs = QOpenGLShader(QOpenGLShader.ShaderTypeBit.Compute)
        if not cs.compileSourceCode(src):
            _nis_emit_shader_log("Shader compute NVScaler: COMPILACIÓN fallida", cs.log())
            self._prog_compute = None
            return False
        compile_note = (cs.log() or "").strip()
        if compile_note and _nis_debug():
            _nis_emit_shader_log("Shader compute NVScaler: log del driver (avisos)", compile_note)

        self._prog_compute = QOpenGLShaderProgram()
        if not self._prog_compute.addShader(cs):
            _nis_emit_shader_log(
                "No se pudo adjuntar el compute al programa",
                self._prog_compute.log(),
            )
            self._prog_compute.removeAllShaders()
            self._prog_compute = None
            return False
        if not self._prog_compute.link():
            _nis_emit_shader_log("Shader compute NVScaler: ENLACE (link) fallido", self._prog_compute.log())
            self._prog_compute.removeAllShaders()
            self._prog_compute = None
            return False
        return True

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
                _nis_emit_shader_log("Blit (vertex+fragment): compilación o enlace fallido", self._prog_blit.log())
            self._prog_blit.removeAllShaders()
            self._prog_blit = None
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

    def _ensure_ubo(self) -> bool:
        if QOpenGLBuffer is None:
            return False
        if self._ubo is not None and self._ubo.isCreated():
            return True
        # Qt6 / PySide6: QOpenGLBuffer.Type no define UniformBuffer (solo Vertex/Index/Pixel*).
        # El nombre de buffer OpenGL es genérico: subimos con un tipo Qt y enlazamos a UBO con
        # glBindBufferBase(GL_UNIFORM_BUFFER, …, bufferId()) en el dispatch.
        self._ubo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._ubo.create()
        return self._ubo.isCreated()

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

    def _set_uniform_int(self, program: QOpenGLShaderProgram, name: str, value: int) -> None:
        loc = _qgl_uniform_location(program, name)
        if loc < 0:
            loc = _qgl_uniform_location_gl(program, name)
        if loc >= 0:
            program.setUniformValue(loc, int(value))

    def _draw_quad_like_blend(self, program: QOpenGLShaderProgram, f: Any) -> bool:
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
        # El UBO usa QOpenGLBuffer(VertexBuffer): bind()/release() del UBO tocan GL_ARRAY_BUFFER
        # y desenlazan el VBO del quad; hay que re-enlazar antes de setAttributeBuffer/draw.
        vb = self._vbo
        if vb is not None:
            vb.bind()
        try:
            program.enableAttributeArray(loc)
            program.setAttributeBuffer(loc, _GL_FLOAT, 0, 4, 0)
            f.glDrawArrays(_GL_TRIANGLES, 0, 6)
            program.disableAttributeArray(loc)
            return True
        finally:
            if use_vao:
                vao.release()

    def _bind_qopengl_widget_backbuffer(
        self,
        gl_widget: QtWidgets.QWidget,
        f: Any,
        ctx: QtGui.QOpenGLContext | None,
    ) -> None:
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
        if prev_fbo is not None and int(prev_fbo) > 0:
            _gl_bind_framebuffer(f, ctx, gl_widget, int(prev_fbo))
            return
        self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)

    def _nis_bind_compute_samplers(
        self, ctx: QtGui.QOpenGLContext, f: Any, read_tid: int | None
    ) -> None:
        _fsr_gl_active_texture_unit0(ctx)
        if read_tid is None:
            self._src_gl_bind_unit0(ctx)
        else:
            f.glBindTexture(_GL_TEXTURE_2D, int(read_tid))
        _fsr_gl_c_active_texture(ctx, _GL_TEXTURE1)
        _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, int(self._tex_coef_s))
        _fsr_gl_c_active_texture(ctx, _GL_TEXTURE2)
        _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, int(self._tex_coef_u))

    def _nis_unbind_compute_samplers(
        self, ctx: QtGui.QOpenGLContext, f: Any, read_tid: int | None
    ) -> None:
        if read_tid is None:
            self._src_gl_unbind_unit0(ctx)
        else:
            f.glBindTexture(_GL_TEXTURE_2D, 0)
        _fsr_gl_c_active_texture(ctx, _GL_TEXTURE1)
        _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
        _fsr_gl_c_active_texture(ctx, _GL_TEXTURE2)
        _fsr_gl_c_bind_texture(ctx, _GL_TEXTURE_2D, 0)
        _fsr_gl_active_texture_unit0(ctx)

    def _log_nis_path_debug(
        self,
        tag: str,
        w: int,
        h: int,
        rw: int,
        rh: int,
        extra: str = "",
    ) -> None:
        if not _nis_debug():
            return
        _nis_log_stderr_hint_once()
        key = (tag, w, h, rw, rh)
        if self._nis_dbg_path_key == key:
            return
        self._nis_dbg_path_key = key
        sx = float(w) / float(max(1, rw))
        sy = float(h) / float(max(1, rh))
        msg = (
            f"{tag}  tex={w}x{h} dst_rect={rw}x{rh}  "
            f"w/dst_w={sx:.4f} h/dst_h={sy:.4f}"
        )
        if extra:
            msg = f"{msg}  ({extra})"
        _nis_dbg(msg)

    def render_gl_in_viewport(
        self,
        gl_widget: QtWidgets.QWidget,
        gv: QtWidgets.QGraphicsView,
    ) -> None:
        if self._frame_bgr is None or QOpenGLShaderProgram is None:
            if _nis_debug() and not getattr(self, "_nis_logged_skip_no_frame", False):
                self._nis_logged_skip_no_frame = True
                _nis_render_trace("salida temprana: sin frame_bgr o QOpenGLShaderProgram")
            return
        if self._gl_failed:
            if _nis_debug() and not getattr(self, "_nis_logged_skip_gl_failed", False):
                self._nis_logged_skip_gl_failed = True
                _nis_render_trace("salida temprana: _gl_failed=True (reinicia preview o toggles NIS)")
            return
        try:
            gl_widget.makeCurrent()
        except Exception as e:
            self._gl_failed = True
            print(f"[NIS] makeCurrent falló (se desactiva reintento GL): {e!r}", flush=True)
            if _nis_debug():
                _nis_dbg(f"makeCurrent failed: {e!r}")
            return
        ctx = gl_widget.context() or QtGui.QOpenGLContext.currentContext()
        if ctx is None:
            if _nis_debug() and not self._nis_dbg_no_ctx_logged:
                _nis_log_stderr_hint_once()
                self._nis_dbg_no_ctx_logged = True
                _nis_dbg(
                    "render skip: sin QOpenGLContext en el viewport (¿preview OpenGL no activo?)"
                )
            return
        _nis_render_trace("makeCurrent OK, contexto listo")
        schedule_khr_debug_install_for_context(ctx)
        log_gl_driver_info_once(ctx)
        xf = _gl_extra_for_context(ctx)
        f = _gl_base_functions(ctx)
        h, w = self._frame_bgr.shape[:2]
        rw0, rh0 = self._item_device_pixel_size_from_view(gv)
        limit_px = nis_max_compute_output_pixels()
        oversize_dst = self._use_compute and (rw0 * rh0 > limit_px)
        direct_blit = (not self._use_compute) or oversize_dst
        _nis_render_trace(
            f"layout tex={w}x{h} dst_item_px={rw0}x{rh0} use_compute={self._use_compute} "
            f"direct_blit={direct_blit} oversize_dst={oversize_dst} limit_px={limit_px}"
        )
        if oversize_dst and not getattr(self, "_nis_oversize_warned", False):
            self._nis_oversize_warned = True
            print(
                f"[NIS] Área destino {rw0}x{rh0} ({rw0 * rh0}px) supera límite compute "
                f"{limit_px}px — solo blit (evita cierres del driver). "
                f"Ajusta VISIOMASTER_NIS_MAX_OUTPUT_PIXELS o reduce el preview.",
                flush=True,
            )

        try:
            if not direct_blit:
                _nis_render_trace("ensure_compute…")
                if not self._ensure_compute():
                    self._gl_failed = True
                    _nis_render_trace("fallo ensure_compute")
                    return
                _nis_render_trace("ensure_compute OK")
            _nis_render_trace("ensure_blit_program…")
            if not self._ensure_blit_program():
                self._gl_failed = True
                _nis_render_trace("fallo ensure_blit_program")
                return
            _nis_render_trace("ensure VBO/VAO…")
            if not self._ensure_vbo() or not self._ensure_vao():
                self._gl_failed = True
                _nis_render_trace("fallo VBO/VAO")
                return
            _nis_render_trace("src_gl_ensure…")
            if not self._src_gl_ensure(ctx, w, h):
                self._gl_failed = True
                _nis_render_trace("fallo src_gl_ensure")
                return
            if self._present_seq != self._upload_applied_seq:
                _nis_render_trace("upload textura vídeo → GPU…")
                rgb = _numpy_bgr_to_rgb_contiguous(self._frame_bgr)
                rgba = _numpy_rgb_to_rgba_contiguous(rgb)
                self._src_gl_upload_rgba(ctx, rgba)
                self._upload_applied_seq = self._present_seq
                _nis_render_trace("upload textura hecho")
            else:
                _nis_render_trace("reutiliza textura (mismo present_seq)")

            vw = max(1, gl_widget.width())
            vh = max(1, gl_widget.height())
            vp_w, vp_h = _gl_viewport_phys_size(gl_widget)
            vbo_key = (vw, vh, self._lay_w, self._lay_h, int(direct_blit), self._viewport_corner_flat(gv))
            rebuild = vbo_key != self._vbo_cache_key or self._cached_verts is None
            if rebuild:
                self._cached_verts = self._build_vertices_from_view(gv, vw, vh)
                self._vbo_cache_key = vbo_key
            verts = self._cached_verts
            assert self._vbo is not None
            self._vbo.bind()
            if rebuild:
                self._vbo.allocate(verts.tobytes(), verts.nbytes)

            f.glViewport(0, 0, vp_w, vp_h)
            f.glDisable(0x0B71)
            f.glDisable(0x0BE2)
            try:
                f.glDisable(0x0C11)
            except Exception:
                pass
            try:
                f.glColorMask(1, 1, 1, 1)
            except Exception:
                pass
            try:
                f.glDisable(_GL_CULL_FACE)
            except Exception:
                pass

            rw, rh = rw0, rh0
            self._preview_overlay_src_wh = (int(w), int(h))
            self._preview_overlay_tgt_wh = (int(rw), int(rh))
            prev_qt_fbo = _gl_query_framebuffer_binding(f, ctx, gl_widget)
            _nis_render_trace(
                f"viewport widget={vw}x{vh} phys={vp_w}x{vp_h} rwrh={rw}x{rh} prev_fbo={prev_qt_fbo!r}"
            )

            if direct_blit:
                extra_blit = (
                    f"dst>{limit_px}px; solo blit"
                    if oversize_dst
                    else "«NIS shaders» desactivado — no se ejecuta NVScaler compute"
                )
                self._log_nis_path_debug(
                    "blit_only" if not oversize_dst else "blit_oversize_cap",
                    w,
                    h,
                    rw,
                    rh,
                    extra_blit,
                )
                self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)
                assert self._prog_blit is not None
                self._prog_blit.bind()
                self._src_gl_bind_unit0(ctx)
                self._set_uniform_int(self._prog_blit, "u_src", 0)
                _nis_render_trace("direct_blit: draw_quad → backbuffer…")
                if not self._draw_quad_like_blend(self._prog_blit, f):
                    self._gl_failed = True
                    self._src_gl_unbind_unit0(ctx)
                    self._prog_blit.release()
                    self._vbo.release()
                    _nis_render_trace("direct_blit: draw_quad falló")
                    return
                self._src_gl_unbind_unit0(ctx)
                self._prog_blit.release()
                _nis_render_trace("direct_blit: OK")
            else:
                ubo_single = pack_nis_scaler_ubo(
                    input_tex_w=w,
                    input_tex_h=h,
                    input_vp_w=w,
                    input_vp_h=h,
                    out_tex_w=rw,
                    out_tex_h=rh,
                    out_vp_w=rw,
                    out_vp_h=rh,
                    sharpness_0_1=self._sharpness_01,
                )
                chain = nis_upscale_chain(w, h, rw, rh)
                use_single = ubo_single is not None and self._ensure_nis_fbo(ctx, rw, rh)
                use_chain = (not use_single) and bool(chain)
                _nis_render_trace(
                    f"compute rama: use_single={use_single} use_chain={use_chain} "
                    f"chain_len={len(chain)}"
                )

                if not use_single and not use_chain:
                    if ubo_single is None:
                        self._log_nis_path_debug(
                            "blit_fallback",
                            w,
                            h,
                            rw,
                            rh,
                            "sin cadena NVScaler (p. ej. downscale: w/dst_w>1); solo blit",
                        )
                    else:
                        self._log_nis_path_debug(
                            "blit_fallback",
                            w,
                            h,
                            rw,
                            rh,
                            "FBO NIS no disponible; blit simple",
                        )
                    self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)
                    assert self._prog_blit is not None
                    self._prog_blit.bind()
                    self._src_gl_bind_unit0(ctx)
                    self._set_uniform_int(self._prog_blit, "u_src", 0)
                    _nis_render_trace("fallback blit (sin single ni chain) draw_quad…")
                    if not self._draw_quad_like_blend(self._prog_blit, f):
                        self._gl_failed = True
                        self._src_gl_unbind_unit0(ctx)
                        self._prog_blit.release()
                        self._vbo.release()
                        _nis_render_trace("fallback blit falló")
                        return
                    self._src_gl_unbind_unit0(ctx)
                    self._prog_blit.release()
                    self._vbo.release()
                    _nis_render_trace("fallback blit OK, return")
                    return
                _nis_render_trace("ensure_coef_textures + UBO…")
                if not self._ensure_coef_textures(ctx):
                    self._gl_failed = True
                    self._vbo.release()
                    _nis_render_trace("fallo ensure_coef_textures")
                    return
                if not self._ensure_ubo():
                    self._gl_failed = True
                    self._vbo.release()
                    _nis_render_trace("fallo ensure_ubo")
                    return
                assert self._ubo is not None
                _nis_render_trace("coef + UBO OK; APIs compute…")

                bind_buf_base = getattr(xf, "glBindBufferBase", None)
                dispatch = getattr(xf, "glDispatchCompute", None)
                bind_img = getattr(xf, "glBindImageTexture", None)
                mem_barrier = getattr(xf, "glMemoryBarrier", None)
                if (
                    bind_buf_base is None
                    or dispatch is None
                    or bind_img is None
                    or mem_barrier is None
                ):
                    _nis_dbg("OpenGL compute API missing; fall back to blit")
                    self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)
                    assert self._prog_blit is not None
                    self._prog_blit.bind()
                    self._src_gl_bind_unit0(ctx)
                    self._set_uniform_int(self._prog_blit, "u_src", 0)
                    if not self._draw_quad_like_blend(self._prog_blit, f):
                        self._gl_failed = True
                        self._src_gl_unbind_unit0(ctx)
                        self._prog_blit.release()
                        self._vbo.release()
                        return
                    self._src_gl_unbind_unit0(ctx)
                    self._prog_blit.release()
                    self._vbo.release()
                    return

                if _nis_debug():
                    _nis_dbg(
                        f"NVScaler dispatch  single={use_single}  tex={w}x{h}  dst={rw}x{rh}  "
                        f"chain_steps={len(chain)}"
                    )

                _fsr1_gl_drain_errors(ctx, "NIS pre compute dispatch")
                _nis_render_trace("post drain_errors pre compute")

                final_tex_id: int
                if use_single:
                    assert ubo_single is not None
                    self._ubo.bind()
                    self._ubo.allocate(ubo_single, len(ubo_single))
                    ubo_id = int(self._ubo.bufferId())
                    self._ubo.release()
                    out_tid = int(self._fbo.texture())
                    bind_buf_base(_GL_UNIFORM_BUFFER, 0, ubo_id)
                    assert self._prog_compute is not None
                    self._prog_compute.bind()
                    self._set_uniform_int(self._prog_compute, "in_texture", 0)
                    self._set_uniform_int(self._prog_compute, "coef_scaler", 1)
                    self._set_uniform_int(self._prog_compute, "coef_usm", 2)
                    self._nis_bind_compute_samplers(ctx, f, None)
                    bind_img(0, out_tid, 0, False, 0, _GL_WRITE_ONLY, _GL_RGBA8)
                    gx = max(1, (rw + _NIS_BLOCK_W - 1) // _NIS_BLOCK_W)
                    gy = max(1, (rh + _NIS_BLOCK_H - 1) // _NIS_BLOCK_H)
                    _nis_render_trace(
                        f"single: glDispatchCompute grid=({gx},{gy},1) out_tex={out_tid} dst={rw}x{rh}"
                    )
                    dispatch(int(gx), int(gy), 1)
                    _nis_render_trace("single: post dispatch → memoryBarrier + textureBarrier")
                    mem_barrier(
                        _GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                        | _GL_TEXTURE_FETCH_BARRIER_BIT
                    )
                    gl_texture_barrier_after_image_write(ctx)
                    _fsr1_gl_drain_errors(ctx, "NIS after compute (single)")
                    _nis_render_trace("single: compute paso terminado")
                    self._nis_unbind_compute_samplers(ctx, f, None)
                    self._prog_compute.release()
                    final_tex_id = int(out_tid)
                else:
                    read_tid: int | None = None
                    cw, ch = w, h
                    primary_out = True
                    for _step_i, (tw, th) in enumerate(chain):
                        _nis_render_trace(
                            f"chain step {_step_i + 1}/{len(chain)}: {tw}x{th} (from {cw}x{ch})"
                        )
                        step_ubo = pack_nis_scaler_ubo(
                            input_tex_w=cw,
                            input_tex_h=ch,
                            input_vp_w=cw,
                            input_vp_h=ch,
                            out_tex_w=tw,
                            out_tex_h=th,
                            out_vp_w=tw,
                            out_vp_h=th,
                            sharpness_0_1=self._sharpness_01,
                        )
                        if step_ubo is None or not self._ensure_nis_fbo_slot(
                            ctx, primary=primary_out, tw=tw, th=th
                        ):
                            self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)
                            assert self._prog_blit is not None
                            self._prog_blit.bind()
                            self._src_gl_bind_unit0(ctx)
                            self._set_uniform_int(self._prog_blit, "u_src", 0)
                            if not self._draw_quad_like_blend(self._prog_blit, f):
                                self._gl_failed = True
                                self._src_gl_unbind_unit0(ctx)
                                self._prog_blit.release()
                                self._vbo.release()
                                return
                            self._src_gl_unbind_unit0(ctx)
                            self._prog_blit.release()
                            self._vbo.release()
                            return
                        out_fbo = self._fbo if primary_out else self._fbo_aux
                        out_tid = int(out_fbo.texture())
                        self._ubo.bind()
                        self._ubo.allocate(step_ubo, len(step_ubo))
                        ubo_id = int(self._ubo.bufferId())
                        self._ubo.release()
                        bind_buf_base(_GL_UNIFORM_BUFFER, 0, ubo_id)
                        assert self._prog_compute is not None
                        self._prog_compute.bind()
                        self._set_uniform_int(self._prog_compute, "in_texture", 0)
                        self._set_uniform_int(self._prog_compute, "coef_scaler", 1)
                        self._set_uniform_int(self._prog_compute, "coef_usm", 2)
                        self._nis_bind_compute_samplers(ctx, f, read_tid)
                        bind_img(0, out_tid, 0, False, 0, _GL_WRITE_ONLY, _GL_RGBA8)
                        gx = max(1, (tw + _NIS_BLOCK_W - 1) // _NIS_BLOCK_W)
                        gy = max(1, (th + _NIS_BLOCK_H - 1) // _NIS_BLOCK_H)
                        _nis_render_trace(
                            f"chain: dispatch grid=({gx},{gy},1) out_tid={out_tid}"
                        )
                        dispatch(int(gx), int(gy), 1)
                        _nis_render_trace(f"chain step {_step_i + 1}: post dispatch barriers")
                        mem_barrier(
                            _GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                            | _GL_TEXTURE_FETCH_BARRIER_BIT
                        )
                        gl_texture_barrier_after_image_write(ctx)
                        _fsr1_gl_drain_errors(
                            ctx, f"NIS after compute chain step {tw}x{th}"
                        )
                        self._nis_unbind_compute_samplers(ctx, f, read_tid)
                        self._prog_compute.release()
                        read_tid = int(out_tid)
                        cw, ch = tw, th
                        primary_out = not primary_out
                    if read_tid is None:
                        self._bind_qopengl_widget_backbuffer(gl_widget, f, ctx)
                        assert self._prog_blit is not None
                        self._prog_blit.bind()
                        self._src_gl_bind_unit0(ctx)
                        self._set_uniform_int(self._prog_blit, "u_src", 0)
                        self._draw_quad_like_blend(self._prog_blit, f)
                        self._src_gl_unbind_unit0(ctx)
                        self._prog_blit.release()
                        self._vbo.release()
                        return
                    final_tex_id = int(read_tid)

                _nis_render_trace(f"unbind image0; blit final_tex_id={final_tex_id} → pantalla")
                _nis_gl_unbind_image_unit0(xf)
                # No usar QOpenGLFramebufferObject.bindDefault() aquí: en QOpenGLWidget el
                # «default» real es defaultFramebufferObject(), no FBO 0; bindDefault()
                # puede colgar o romper el contexto y dejar el preview congelado.
                gl_texture_barrier_after_image_write(ctx)
                _fsr1_gl_drain_errors(ctx, "NIS pre blit to screen")
                _nis_render_trace("restore Qt FBO + viewport + prog_blit para quad final")
                self._restore_qt_draw_framebuffer(f, gl_widget, prev_qt_fbo, ctx)
                f.glViewport(0, 0, vp_w, vp_h)
                assert self._prog_blit is not None
                self._prog_blit.bind()
                _fsr_gl_active_texture_unit0(ctx)
                f.glBindTexture(_GL_TEXTURE_2D, final_tex_id)
                self._set_uniform_int(self._prog_blit, "u_src", 0)
                _nis_render_trace("draw_quad FBO→backbuffer…")
                if not self._draw_quad_like_blend(self._prog_blit, f):
                    self._gl_failed = True
                    f.glBindTexture(_GL_TEXTURE_2D, 0)
                    self._prog_blit.release()
                    self._vbo.release()
                    _nis_render_trace("quad final falló")
                    return
                f.glBindTexture(_GL_TEXTURE_2D, 0)
                self._prog_blit.release()
                _fsr1_gl_drain_errors(ctx, "NIS after blit to screen")
                _nis_render_trace("quad final OK")

            self._vbo.release()
            if not self._compute_ok_logged and not direct_blit:
                self._compute_ok_logged = True
                print(
                    "[NIS] Preview NVScaler (compute) active. "
                    "Si el preview es más pequeño que el vídeo, se usa blit simple. "
                    "Más detalle: VISIOMASTER_DEBUG_NIS=1 y consola visible.",
                    flush=True,
                )
                if _nis_debug():
                    _nis_dbg(
                        f"NVScaler compute OK  tex={w}x{h} dst_rect={rw}x{rh} "
                        f"(mismo criterio que el overlay source→target)"
                    )
            _nis_render_trace("render_gl_in_viewport fin OK (frame)")
        except Exception as e:
            self._gl_failed = True
            try:
                vb = self._vbo
                if vb is not None:
                    vb.release()
            except Exception:
                pass
            print(f"[NIS] render_gl_in_viewport error (se desactiva reintento GL): {e!r}", flush=True)
            traceback.print_exc()
            if _nis_debug():
                _nis_dbg(f"render_gl_in_viewport exception: {e!r}")

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
        self._compute_ok_logged = False
        self._upload_applied_seq = -1
        self._vbo_cache_key = None
        self._cached_verts = None
        self._preview_overlay_src_wh = None
        self._preview_overlay_tgt_wh = None
        self._nis_dbg_path_key = None
        self._nis_dbg_no_ctx_logged = False
        self._nis_oversize_warned = False
        self._nis_logged_skip_no_frame = False
        self._nis_logged_skip_gl_failed = False
        self._destroy_gl_objects()
