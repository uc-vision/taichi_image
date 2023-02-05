from taichi_image.util import cache
import taichi as ti
import taichi.math as tm
from taichi_image import types

from typeguard import typechecked


@ti.func
def index_clamped(src: ti.template(), idx: tm.ivec2):
  return src[tm.clamp(idx, 0, tm.ivec2(src.shape) - 1)]

@ti.func
def sample_bilinear(src: ti.template(), p: tm.vec2):
  p1 = ti.cast(p, ti.i32)

  frac = p - ti.cast(p1, ti.f32)
  y1 = tm.mix(index_clamped(src, p1), 
              index_clamped(src, p1 + tm.ivec2(1, 0)),
              frac.x)
  y2 = tm.mix(index_clamped(src, p1 + tm.ivec2(0, 1)),
              index_clamped(src, p1 + tm.ivec2(1, 1)), 
              frac.x)
  return tm.mix(y1, y2, frac.y)

@cache    
@typechecked
def bilinear_kernel(in_dtype=ti.u8, out_dtype=None):
  if out_dtype is None:
    out_dtype = in_dtype
  
  in_vec3 = ti.types.vector(3, out_dtype)
  out_vec3 = ti.types.vector(3, out_dtype)


  intensity_scale = types.pixel_types[out_dtype] / types.pixel_types[in_dtype]

  @ti.kernel
  def f(src: ti.types.ndarray(dtype=in_vec3, ndim=2), 
        dst: ti.types.ndarray(dtype=out_vec3, ndim=2),
        scale: tm.vec2):
    

    for I in ti.grouped(dst):
      p = ti.cast(I, ti.f32) / scale
      dst[I] = ti.cast(sample_bilinear(src, p) * intensity_scale, out_dtype)

  return f

def resize_bilinear(src, size, scale=None, dtype=None):
  
  if dtype is None:
    dtype = types.ti_type(src)

  if scale is None:
    scale = tm.vec2(size) / tm.vec2(src.shape[:2])

  dst = types.zeros_array(src, (size[1], size[0], 3), dtype)
  f = bilinear_kernel(types.ti_type(src), dtype)
  f(src, dst, tm.vec2(scale))
  return dst

def scale_bilinear(src, scale, dtype=None):

  h, w = src.shape[:2]
  size = tm.vec2(w, h) * scale
  return resize_bilinear(src, tm.ivec2(size), scale, dtype)
  
