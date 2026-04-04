"""
OpenGL mix of two BGR video frames for preview (QGraphicsScene item).

Used when Settings → Frame Interpolation → method Linear (GPU) is active (video preview).
Requires QGraphicsView viewport to be a QOpenGLWidget.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from PySide6.QtOpenGL import (
        QOpenGLBuffer,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLTexture,
    )
except ImportError:  # pragma: no cover
    QOpenGLBuffer = None  # type: ignore[misc, assignment]
    QOpenGLShader = None  # type: ignore[misc, assignment]
    QOpenGLShaderProgram = None  # type: ignore[misc, assignment]
    QOpenGLTexture = None  # type: ignore[misc, assignment]

# OpenGL ES / desktop constants (avoid missing QOpenGL module on some PySide6 builds)
_GL_FLOAT = 0x1406
_GL_TRIANGLES = 0x0004
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2

# Bump when changing shader source so existing sessions re-link (uniforms / logic).
_BLEND_SHADER_PROGRAM_VERSION = 2


_VS = """#version 330 core
layout(location = 0) in vec4 a_pos_uv;
out vec2 v_uv;
void main() {
  gl_Position = vec4(a_pos_uv.xy, 0.0, 1.0);
  v_uv = a_pos_uv.zw;
}
"""

_FS = """#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform float u_w;
uniform int u_mode;
out vec4 fragColor;

float lumRgb(vec3 c) {
  return dot(clamp(c, 0.0, 1.0), vec3(0.299, 0.587, 0.114));
}

vec3 toLin(vec3 c) {
  return pow(max(c, vec3(1e-6)), vec3(2.2));
}

vec3 toSrgb(vec3 c) {
  return pow(max(c, vec3(1e-6)), vec3(1.0 / 2.2));
}

float edgeStrength(vec2 tuv, vec2 d) {
  float xp = lumRgb(texture(u_tex0, tuv + vec2(d.x, 0.0)).rgb);
  float xm = lumRgb(texture(u_tex0, tuv - vec2(d.x, 0.0)).rgb);
  float yp = lumRgb(texture(u_tex0, tuv + vec2(0.0, d.y)).rgb);
  float ym = lumRgb(texture(u_tex0, tuv - vec2(0.0, d.y)).rgb);
  float gx = xp - xm;
  float gy = yp - ym;
  return smoothstep(0.03, 0.22, sqrt(gx * gx + gy * gy));
}

void main() {
  vec2 tuv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec3 c0 = texture(u_tex0, tuv).rgb;
  vec3 c1 = texture(u_tex1, tuv).rgb;
  float w = clamp(u_w, 0.0, 1.0);
  int m = u_mode;
  if (m < 0 || m > 4) {
    m = 0;
  }

  if (m == 0) {
    fragColor = vec4(mix(c0, c1, w), 1.0);
    return;
  }

  vec3 L0 = toLin(c0);
  vec3 L1 = toLin(c1);
  ivec2 ts = textureSize(u_tex0, 0);
  vec2 d = 1.0 / max(vec2(float(ts.x), float(ts.y)), vec2(1.0));

  if (m == 1) {
    fragColor = vec4(toSrgb(mix(L0, L1, w)), 1.0);
    return;
  }

  float l0 = lumRgb(c0);
  float l1 = lumRgb(c1);
  float denom = max((1.0 - w) * (l0 + 0.04) + w * (l1 + 0.04), 1e-4);
  float w_lm = (w * (l1 + 0.04)) / denom;

  if (m == 2) {
    fragColor = vec4(toSrgb(mix(L0, L1, w_lm)), 1.0);
    return;
  }

  float ed = edgeStrength(tuv, d);

  if (m == 3) {
    float we = mix(w, 0.5, ed * 0.55);
    fragColor = vec4(toSrgb(mix(L0, L1, clamp(we, 0.0, 1.0))), 1.0);
    return;
  }

  float wep = mix(w_lm, 0.5, ed * 0.48);
  fragColor = vec4(toSrgb(mix(L0, L1, clamp(wep, 0.0, 1.0))), 1.0);
}
"""


def _numpy_bgr_to_rgb_contiguous(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 BGR")
    return np.ascontiguousarray(arr[..., ::-1])


class VideoBlendOpenGLItem(QtWidgets.QGraphicsObject):
    """Draws mix(prev, curr, w) with a fragment shader inside an OpenGL QGraphicsView viewport."""

    def __init__(self) -> None:
        super().__init__()
        self._prev: np.ndarray | None = None
        self._curr: np.ndarray | None = None
        self._w: float = 0.0
        self._bw: int = 1
        self._bh: int = 1
        self._program: QOpenGLShaderProgram | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._tex0: QOpenGLTexture | None = None
        self._tex1: QOpenGLTexture | None = None
        self._tex_staging0: np.ndarray | None = None
        self._tex_staging1: np.ndarray | None = None
        self._gl_failed: bool = False
        self._blend_mode: int = 0
        self.setZValue(1.0)

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802
        return QtCore.QRectF(0.0, 0.0, float(self._bw), float(self._bh))

    def set_blend_frames(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
        w: float,
        blend_mode: int = 0,
    ) -> None:
        self._prev = prev_bgr
        self._curr = curr_bgr
        self._w = float(w)
        self._blend_mode = int(max(0, min(4, blend_mode)))
        h, wi = curr_bgr.shape[:2]
        if wi != self._bw or h != self._bh:
            self.prepareGeometryChange()
            self._bw = int(wi)
            self._bh = int(h)
        self.update()

    def _destroy_gl_objects(self) -> None:
        if self._tex0 is not None:
            self._tex0.destroy()
            self._tex0 = None
        if self._tex1 is not None:
            self._tex1.destroy()
            self._tex1 = None
        if self._vbo is not None:
            self._vbo.destroy()
            self._vbo = None
        if self._program is not None:
            self._program.removeAllShaders()
            self._program = None

    def _ensure_program(self) -> bool:
        if QOpenGLShaderProgram is None or QOpenGLShader is None:
            return False
        if self._program is not None and self._program.isLinked():
            if getattr(self, "_blend_prog_ver", 0) == _BLEND_SHADER_PROGRAM_VERSION:
                return True
            self._program.removeAllShaders()
            self._program = None
        self._program = QOpenGLShaderProgram()
        ok = self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _VS)
        ok = ok and self._program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _FS
        )
        ok = ok and self._program.link()
        if not ok:
            self._program.removeAllShaders()
            self._program = None
            return False
        self._blend_prog_ver = _BLEND_SHADER_PROGRAM_VERSION
        return True

    def _ensure_vbo(self) -> bool:
        if QOpenGLBuffer is None:
            return False
        if self._vbo is not None:
            return True
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vbo.create()
        return self._vbo.isCreated()

    def _ensure_texture(self, tex: QOpenGLTexture | None, w: int, h: int) -> QOpenGLTexture | None:
        if QOpenGLTexture is None:
            return None
        reuse = False
        if tex is not None:
            try:
                reuse = int(tex.width()) == w and int(tex.height()) == h
            except RuntimeError:
                reuse = False
            if not reuse:
                try:
                    tex.destroy()
                except RuntimeError:
                    pass
                tex = None
        if reuse:
            return tex
        t = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        t.setSize(w, h)
        t.setFormat(QOpenGLTexture.TextureFormat.RGB8_UNorm)
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
        return t

    def _upload_tex(self, tex: QOpenGLTexture, rgb: np.ndarray) -> None:
        """Upload RGB8 without QImage::setData (avoids Qt reconfiguring format/mips on allocated storage)."""
        h, w = rgb.shape[:2]
        tex.bind(0)
        tex.setData(
            0,
            0,
            0,
            w,
            h,
            1,
            QOpenGLTexture.PixelFormat.RGB,
            QOpenGLTexture.PixelType.UInt8,
            rgb.tobytes(),
        )
        tex.release()

    def _build_vertices(self, painter: QtGui.QPainter) -> np.ndarray:
        """Two triangles (TL,TR,BL) and (TR,BR,BL) in NDC with UVs."""
        rect = self.boundingRect()
        dt = painter.deviceTransform()
        dev = painter.device()
        vw = max(1, dev.width())
        vh = max(1, dev.height())
        ndc_uv = []
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            p = dt.map(QtCore.QPointF(cx, cy))
            x_ndc = 2.0 * float(p.x()) / float(vw) - 1.0
            y_ndc = 1.0 - 2.0 * float(p.y()) / float(vh)
            ndc_uv.append((x_ndc, y_ndc))
        tl, tr, bl, br = ndc_uv
        verts = np.array(
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
        return verts

    def paint(  # noqa: N802
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None,
    ) -> None:
        del option
        if (
            self._prev is None
            or self._curr is None
            or QOpenGLShaderProgram is None
            or widget is None
        ):
            return
        from PySide6.QtOpenGLWidgets import QOpenGLWidget

        if not isinstance(widget, QOpenGLWidget):
            return
        if self._gl_failed:
            return
        ctx = QtGui.QOpenGLContext.currentContext()
        if ctx is None:
            return

        try:
            painter.beginNativePainting()
            if not self._ensure_program() or not self._ensure_vbo():
                self._gl_failed = True
                return
            r0 = _numpy_bgr_to_rgb_contiguous(self._prev)
            r1 = _numpy_bgr_to_rgb_contiguous(self._curr)
            if r0.shape != r1.shape:
                return
            h, w = r1.shape[:2]
            self._tex_staging0 = r0
            self._tex_staging1 = r1

            self._tex0 = self._ensure_texture(self._tex0, w, h)
            self._tex1 = self._ensure_texture(self._tex1, w, h)
            if self._tex0 is None or self._tex1 is None:
                self._gl_failed = True
                return
            self._upload_tex(self._tex0, r0)
            self._upload_tex(self._tex1, r1)

            verts = self._build_vertices(painter)
            assert self._vbo is not None
            self._vbo.bind()
            self._vbo.allocate(verts.tobytes(), verts.nbytes)

            dev = painter.device()
            f = ctx.functions()
            f.glViewport(0, 0, max(1, dev.width()), max(1, dev.height()))
            f.glDisable(_GL_DEPTH_TEST)
            f.glDisable(_GL_BLEND)

            assert self._program is not None
            self._program.bind()
            self._tex0.bind(0)
            self._tex1.bind(1)
            self._program.setUniformValue("u_tex0", 0)
            self._program.setUniformValue("u_tex1", 1)
            self._program.setUniformValue("u_w", float(max(0.0, min(1.0, self._w))))
            self._program.setUniformValue("u_mode", int(self._blend_mode))

            loc = self._program.attributeLocation("a_pos_uv")
            self._program.enableAttributeArray(loc)
            self._program.setAttributeBuffer(loc, _GL_FLOAT, 0, 4, 0)
            f.glDrawArrays(_GL_TRIANGLES, 0, 6)
            self._program.disableAttributeArray(loc)

            self._tex0.release()
            self._tex1.release()
            self._program.release()
            self._vbo.release()
        except Exception:
            self._gl_failed = True
        finally:
            painter.endNativePainting()

    def reset_gl_state(self) -> None:
        self._gl_failed = False
        self._destroy_gl_objects()
