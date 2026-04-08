// AMD FidelityFX Super Resolution v1.0.2 — EASU pass (RGB, luma-directed).
// SPDX: MIT (AMD copyright below). Adapted from public mpv/GPUOpen ports for VisoMaster Fusion preview.
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

#version 330 core

uniform sampler2D u_src;
uniform vec2 u_inSize;
uniform vec2 u_outSize;

layout(location = 0) out vec4 fragColor;

#define FSR_EASU_DERING 1
#define FSR_EASU_SIMPLE_ANALYSIS 1
#define FSR_EASU_QUIT_EARLY 0
#define FSR_EASU_DIR_THRESHOLD 32768.0

float APrxLoRcpF1(float a) {
  return uintBitsToFloat(uint(0x7ef07ebb) - floatBitsToUint(a));
}
float APrxLoRsqF1(float a) {
  return uintBitsToFloat(uint(0x5f347d74) - (floatBitsToUint(a) >> uint(1)));
}
float AMin3F1(float x, float y, float z) { return min(x, min(y, z)); }
float AMax3F1(float x, float y, float z) { return max(x, max(y, z)); }

float lumRgb(vec3 c) {
  return dot(clamp(c, 0.0, 1.0), vec3(0.299, 0.587, 0.114));
}

void FsrEasuTap(
    inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len,
    float lob, float clp, vec3 c) {
  vec2 v;
  v.x = (off.x * dir.x) + (off.y * dir.y);
  v.y = (off.x * (-dir.y)) + (off.y * dir.x);
  v *= len;
  float d2 = v.x * v.x + v.y * v.y;
  d2 = min(d2, clp);
  float wB = float(2.0 / 5.0) * d2 + -1.0;
  float wA = lob * d2 + -1.0;
  wB *= wB;
  wA *= wA;
  wB = float(25.0 / 16.0) * wB + float(-(25.0 / 16.0 - 1.0));
  float w = wB * wA;
  aC += c * w;
  aW += w;
}

void FsrEasuSet(
    inout vec2 dir, inout float len, vec2 pp,
    float b, float c, float i, float j, float f, float e,
    float k, float l, float h, float g, float o, float n) {
  vec4 w = vec4(0.0);
  w.x = (1.0 - pp.x) * (1.0 - pp.y);
  w.y = pp.x * (1.0 - pp.y);
  w.z = (1.0 - pp.x) * pp.y;
  w.w = pp.x * pp.y;
  float lA = dot(w, vec4(b, c, f, g));
  float lB = dot(w, vec4(e, f, i, j));
  float lC = dot(w, vec4(f, g, j, k));
  float lD = dot(w, vec4(g, h, k, l));
  float lE = dot(w, vec4(j, k, n, o));
  float dc = lD - lC;
  float cb = lC - lB;
  float lenX = max(abs(dc), abs(cb));
  lenX = APrxLoRcpF1(lenX);
  float dirX = lD - lB;
  lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
  lenX *= lenX;
  float ec = lE - lC;
  float ca = lC - lA;
  float lenY = max(abs(ec), abs(ca));
  lenY = APrxLoRcpF1(lenY);
  float dirY = lE - lA;
  lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
  lenY *= lenY;
  len = lenX + lenY;
  dir = vec2(dirX, dirY);
}

vec3 samp(vec2 fp, vec2 hp) {
  vec2 p = fp + hp;
  vec2 uv = p / u_inSize;
  uv.y = 1.0 - uv.y;
  return texture(u_src, uv).rgb;
}

void main() {
  vec2 op = gl_FragCoord.xy;
  vec2 pp = op * u_inSize / u_outSize - vec2(0.5);
  vec2 fp = floor(pp);
  pp -= fp;

  vec3 b = samp(fp, vec2(0.5, -0.5));
  vec3 c = samp(fp, vec2(1.5, -0.5));
  vec3 e = samp(fp, vec2(-0.5, 0.5));
  vec3 f = samp(fp, vec2(0.5, 0.5));
  vec3 g = samp(fp, vec2(1.5, 0.5));
  vec3 h = samp(fp, vec2(2.5, 0.5));
  vec3 i = samp(fp, vec2(-0.5, 1.5));
  vec3 j = samp(fp, vec2(0.5, 1.5));
  vec3 k = samp(fp, vec2(1.5, 1.5));
  vec3 l = samp(fp, vec2(2.5, 1.5));
  vec3 n = samp(fp, vec2(0.5, 2.5));
  vec3 o = samp(fp, vec2(1.5, 2.5));

  float bL = lumRgb(b);
  float cL = lumRgb(c);
  float iL = lumRgb(i);
  float jL = lumRgb(j);
  float fL = lumRgb(f);
  float eL = lumRgb(e);
  float kL = lumRgb(k);
  float lL = lumRgb(l);
  float hL = lumRgb(h);
  float gL = lumRgb(g);
  float oL = lumRgb(o);
  float nL = lumRgb(n);

  vec2 dir = vec2(0.0);
  float len = 0.0;
  FsrEasuSet(dir, len, pp, bL, cL, iL, jL, fL, eL, kL, lL, hL, gL, oL, nL);

  vec2 dir2 = dir * dir;
  float dirR = dir2.x + dir2.y;
  bool zro = dirR < float(1.0 / FSR_EASU_DIR_THRESHOLD);
  dirR = APrxLoRsqF1(dirR);
#if (FSR_EASU_QUIT_EARLY == 1)
  if (zro) {
    vec4 w = vec4(0.0);
    w.x = (1.0 - pp.x) * (1.0 - pp.y);
    w.y = pp.x * (1.0 - pp.y);
    w.z = (1.0 - pp.x) * pp.y;
    w.w = pp.x * pp.y;
    vec3 ocol = w.x * f + w.y * g + w.z * j + w.w * k;
    fragColor = vec4(clamp(ocol, 0.0, 1.0), 1.0);
    return;
  }
#else
  dirR = zro ? 1.0 : dirR;
  dir.x = zro ? 1.0 : dir.x;
#endif
  dir *= vec2(dirR);
  len = len * 0.5;
  len *= len;
  float stretch = (dir.x * dir.x + dir.y * dir.y) * APrxLoRcpF1(max(abs(dir.x), abs(dir.y)));
  vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 + -0.5 * len);
  float lob = 0.5 + float((1.0 / 4.0 - 0.04) - 0.5) * len;
  float clp = APrxLoRcpF1(lob);

  vec3 aC = vec3(0.0);
  float aW = 0.0;
  FsrEasuTap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, b);
  FsrEasuTap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, c);
  FsrEasuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, i);
  FsrEasuTap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, j);
  FsrEasuTap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, f);
  FsrEasuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, e);
  FsrEasuTap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, k);
  FsrEasuTap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, l);
  FsrEasuTap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, h);
  FsrEasuTap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, g);
  FsrEasuTap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, o);
  FsrEasuTap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, n);

  vec3 pix = aC / max(aW, 1e-5);
#if (FSR_EASU_DERING == 1)
  vec3 min1 = min(min(min(f, g), j), k);
  vec3 max1 = max(max(max(f, g), j), k);
  pix = clamp(pix, min1, max1);
#endif
  fragColor = vec4(clamp(pix, 0.0, 1.0), 1.0);
}
