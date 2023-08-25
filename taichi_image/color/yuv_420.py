from functools import lru_cache

import taichi as ti
import taichi.math as tm


from taichi_image import types

cache = lru_cache(maxsize=None)


YCrCb_T_bgr = tm.mat3(
  0.299, 0.587, 0.114,
  -0.168736, -0.331264, 0.5,
  0.5, -0.418688, -0.081312
)

bgr_T_YCrCb = YCrCb_T_bgr.inverse()

@ti.func
def bgr_YCrCb(bgr:tm.vec3):
  return YCrCb_T_bgr @ bgr + tm.vec3(0, 0.5, 0.5)

@ti.func
def rgb_YCrCb(rgb:tm.vec3):
  return bgr_YCrCb(rgb.bgr)

@ti.func
def YCrCb_bgr(YCrCb:tm.vec3):
  return bgr_T_YCrCb @ (YCrCb - tm.vec3(0, 0.5, 0.5))

@ti.func
def YCrCb_rgb(YCrCb:tm.vec3):
  return YCrCb_bgr(YCrCb).bgr



@cache
def rgb_yuv420_kernel(in_dtype, out_dtype=None):
  vec3 = ti.types.vector(3, in_dtype)
  out_dtype = out_dtype or in_dtype

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

@cache
def yuv420_rgb_kernel(in_dtype, out_dtype=None):
  out_dtype = out_dtype or in_dtype
  vec3 = ti.types.vector(3, out_dtype)

  in_scale = types.scale_factor[in_dtype]
  out_scale = types.scale_factor[out_dtype]


  @ti.kernel
  def f(y_image: ti.types.ndarray(dtype=in_dtype, ndim=2),
        uv_image: ti.types.ndarray(dtype=in_dtype, ndim=3),
        rgb_image: ti.types.ndarray(dtype=vec3, ndim=2)):
    
    ti.loop_config(block_dim=512)
    for I in ti.grouped(rgb_image):

      yuv = vec3(y_image[I],
        uv_image[1, I.x // 2, I.y // 2],
        uv_image[0, I.x // 2, I.y // 2])

      rgb = YCrCb_rgb(yuv / in_scale)
      rgb_image[I] = ti.cast(tm.clamp(0, 1, rgb) * out_scale, out_dtype)

  return f


def split_yuv_420(yuv):
  height = yuv.shape[0] * 2 // 3
  width = yuv.shape[1]

  y = yuv[:height]
  uv = yuv[height:].reshape(2, height//2, width//2)

  return y, uv, (width, height)


def rgb_yuv420_image(src, dtype=None):
  if dtype is None:
    dtype = types.ti_type(src)

  height, width, _ = src.shape

  # yuv = types.empty_like(src, ((height * 3) // 2, width), dtype)
  yuv = types.zeros_like(src, ((height * 3) // 2, width), dtype)
  y, uv, _ = split_yuv_420(yuv)


  f = rgb_yuv420_kernel(types.ti_type(src), dtype)
  f(src, y, uv)
  
  return yuv

def yuv420_rgb_image(yuv, dtype=None):
  if dtype is None:
    dtype = types.ti_type(yuv)

  y, uv, (w, h) = split_yuv_420(yuv)
  rgb = types.zeros_like(yuv, (h, w, 3), dtype)

  f = yuv420_rgb_kernel(dtype, types.ti_type(rgb))
  f(y, uv, rgb)

  return rgb
  

