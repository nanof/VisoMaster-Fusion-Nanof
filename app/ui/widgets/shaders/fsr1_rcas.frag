// AMD FidelityFX Super Resolution v1.0.2 — RCAS pass (RGB).
// SPDX: MIT (AMD copyright below). Adapted for VisoMaster Fusion preview.
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

uniform sampler2D u_tex;
uniform vec2 u_outSize;
uniform float u_sharpness;

layout(location = 0) out vec4 fragColor;

#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))
#define FSR_RCAS_DENOISE 1

float APrxMedRcpF1(float a) {
  float b = uintBitsToFloat(uint(0x7ef19fff) - floatBitsToUint(a));
  return b * (-b * a + 2.0);
}
float AMax3F1(float x, float y, float z) { return max(x, max(y, z)); }
float AMin3F1(float x, float y, float z) { return min(x, min(y, z)); }

void main() {
  vec2 tc = (gl_FragCoord.xy + vec2(0.5)) / u_outSize;
  tc.y = 1.0 - tc.y;
  vec2 px = vec2(1.0) / u_outSize;

  vec3 b = texture(u_tex, tc + vec2(0.0, -px.y)).rgb;
  vec3 d = texture(u_tex, tc + vec2(-px.x, 0.0)).rgb;
  vec3 e = texture(u_tex, tc).rgb;
  vec3 f = texture(u_tex, tc + vec2(px.x, 0.0)).rgb;
  vec3 h = texture(u_tex, tc + vec2(0.0, px.y)).rgb;

  vec3 mn1 = min(min(min(b, d), f), h);
  vec3 mx1 = max(max(max(b, d), f), h);
  vec2 peakC = vec2(1.0, -4.0);

  vec3 hitMinL = min(mn1, e) / (4.0 * mx1 + vec3(1e-5));
  vec3 hitMaxL = (vec3(peakC.x) - max(mx1, e)) / (4.0 * mn1 + vec3(peakC.y));
  vec3 lobeL = max(-hitMinL, hitMaxL);
  float sharp = clamp(u_sharpness, 0.0, 2.0);
  vec3 lobe = max(vec3(-FSR_RCAS_LIMIT), min(lobeL, vec3(0.0))) * exp2(-sharp);

#if (FSR_RCAS_DENOISE == 1)
  vec3 nzv = 0.25 * b + 0.25 * d + 0.25 * f + 0.25 * h - e;
  vec3 mx3 = max(max(max(b, d), e), max(f, h));
  vec3 mn3 = min(min(min(b, d), e), min(f, h));
  vec3 nz = clamp(abs(nzv) * vec3(
      APrxMedRcpF1(max(max(b.r, d.r), max(e.r, max(f.r, h.r))) - min(min(b.r, d.r), min(e.r, min(f.r, h.r)))),
      APrxMedRcpF1(max(max(b.g, d.g), max(e.g, max(f.g, h.g))) - min(min(b.g, d.g), min(e.g, min(f.g, h.g)))),
      APrxMedRcpF1(max(max(b.b, d.b), max(e.b, max(f.b, h.b))) - min(min(b.b, d.b), min(e.b, min(f.b, h.b))))),
      0.0, 1.0);
  nz = -0.5 * nz + 1.0;
  lobe *= nz;
#endif

  vec3 rcpL = vec3(
      APrxMedRcpF1(4.0 * lobe.r + 1.0),
      APrxMedRcpF1(4.0 * lobe.g + 1.0),
      APrxMedRcpF1(4.0 * lobe.b + 1.0));
  vec3 pix = (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL;
  fragColor = vec4(clamp(pix, 0.0, 1.0), 1.0);
}
