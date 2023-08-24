from functools import lru_cache

import taichi as ti
import taichi.math as tm
import numpy as np

from taichi_image import types

cache = lru_cache(maxsize=None)


@ti.func
def rgb_gray(rgb) -> ti.f32:
  # 0.299⋅R+0.587⋅G+0.114⋅B
  return tm.dot(rgb, tm.vec3(0.299, 0.587, 0.114))

@ti.func
def bgr_gray(bgr) -> ti.f32:
  # 0.114⋅B+0.587⋅G+0.299⋅R
  return tm.dot(bgr, tm.vec3(0.114, 0.587, 0.299))

def rgb_linear(rgb):
  return ti.select(rgb <= 0.04045, 
     rgb / 12.92,
    tm.pow((rgb + 0.055) / 1.055, 2.4))

@ti.func
def rgb_ciexyz(rgb:tm.vec3):
  linear = rgb_linear(rgb)
  m = tm.mat3x3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
  )
  return m @ linear


@ti.func
def bgr_YCrCb(bgr:tm.vec3):
  m = tm.mat3(
    0.299, 0.587, 0.114,
    -0.168736, -0.331264, 0.5,
    0.5, -0.418688, -0.081312
  )
  return m @ bgr + tm.vec3(0, 0.5, 0.5)


@ti.func
def rgb_YCrCb(rgb:tm.vec3):
  return bgr_YCrCb(rgb.bgr)


@ti.func
def YCrCb_bgr(YCrCb:tm.vec3):
  m = tm.mat3(
    1, 0, 1.402,
    1, -0.344136, -0.714136,
    1, 1.772, 0
  )
  return m @ (YCrCb - tm.vec3(0, 0.5, 0.5))

@ti.func
def YCrCb_rgb(YCrCb:tm.vec3):
  return YCrCb_bgr(YCrCb).bgr

@cache
def rgb_yuv420_kernel(in_dtype, out_dtype=None):
  vec3 = ti.types.vector(3, in_dtype)
  out_dtype = in_dtype or None

  in_scale = types.scale_factor[in_dtype]
  out_scale = types.scale_factor[out_dtype]


  @ti.kernel
  def f(src: ti.types.ndarray(dtype=vec3, ndim=2), 
        y_image: ti.types.ndarray(dtype=out_dtype, ndim=2),
        uv_image: ti.types.ndarray(dtype=out_dtype, ndim=3)):
    
    ti.loop_config(block_dim=512)
    for I in ti.grouped(ti.ndrange(uv_image.shape[1], uv_image.shape[2])):
      p = I * 2
      uv = tm.vec2(0.0)

      for offset in ti.static(ti.ndrange(2, 2)):
        yuv = rgb_YCrCb(src[p + offset] / in_scale)
        y_image[p + offset] = ti.cast(tm.clamp(0, 1, yuv.x) * out_scale, out_dtype)
        uv += yuv.yz

      out_uv = ti.cast(tm.clamp(0, 1, (uv / 4.0)) * out_scale, out_dtype)
      uv_image[1, I.x, I.y] = out_uv.x
      uv_image[0, I.x, I.y] = out_uv.y

  return f


def rgb_yuv420_image(src, dtype=None):
  if dtype is None:
    dtype = types.ti_type(src)

  height, width, _ = src.shape

  # yuv = types.empty_like(src, ((height * 3) // 2, width), dtype)
  yuv = types.zeros_like(src, ((height * 3) // 2, width), dtype)

  y = yuv[:height]
  uv = yuv[height:].reshape(2, height//2, width//2)

  f = rgb_yuv420_kernel(types.ti_type(src), dtype)
  f(src, y, uv)
  
  return yuv