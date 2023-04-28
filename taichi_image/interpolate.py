from enum import Enum
from taichi_image.util import cache
import taichi as ti
import taichi.math as tm
from taichi_image import types

from typeguard import typechecked

class ImageTransform(Enum):
  none = 'none'
  rotate_90 = 'rotate_90'
  rotate_180 = 'rotate_180'
  rotate_270 = 'rotate_270'
  transpose = 'transpose'
  flip_horiz = 'flip_horiz'
  flip_vert = 'flip_vert'
  transverse = 'transverse'

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

@ti.func
def transformed(shape:tm.ivec2, p:tm.ivec2, transform:ti.template()):
  
  if ti.static(transform == ImageTransform.rotate_90):
    return tm.ivec2(shape.y - p.y - 1, p.x)
  elif ti.static(transform == ImageTransform.rotate_180):
    return tm.ivec2(shape.x - p.x - 1, shape.y - p.y - 1)
  elif ti.static(transform == ImageTransform.rotate_270):
    return tm.ivec2(p.y, shape.x - p.x - 1)
  elif ti.static(transform == ImageTransform.transpose):
    return tm.ivec2(p.y, p.x)
  elif ti.static(transform == ImageTransform.flip_vert):
    return tm.ivec2(shape.x - p.x - 1, p.y)
  elif ti.static(transform == ImageTransform.flip_horiz):
    return tm.ivec2(p.x, shape.y - p.y - 1)
  elif ti.static(transform == ImageTransform.transverse):
    return tm.ivec2(shape.y - p.y - 1, shape.x - p.x - 1)
  else:    
    return p


@ti.func
def bilinear_func(src: ti.template(), dst: ti.template(), 
                  scale: ti.f32, intensity_scale: ti.f32,  transform:ti.template(), out_dtype:ti.template()):
  
  for I in ti.grouped(dst):
    p = ti.cast(transformed(tm.ivec2(dst.shape), I, transform), ti.f32) / scale
    # p = ti.cast(I, ti.f32) / scale
    dst[I] = ti.cast(sample_bilinear(src, p) * intensity_scale, out_dtype)



@cache    
def bilinear_kernel(in_dtype=ti.u8, out_dtype=None):
  if out_dtype is None:
    out_dtype = in_dtype
  
  in_vec3 = ti.types.vector(3, out_dtype)
  out_vec3 = ti.types.vector(3, out_dtype)

  intensity_scale = types.scale_factor[out_dtype] / types.scale_factor[in_dtype]

  @ti.kernel
  def f(src: ti.types.ndarray(dtype=in_vec3, ndim=2), 
        dst: ti.types.ndarray(dtype=out_vec3, ndim=2),
        scale: tm.vec2, transform:ti.template()):
    bilinear_func(src, dst, scale, intensity_scale, transform, out_dtype)
  return f


def transformed_size(size, transform:ImageTransform):
  w, h = size
  if transform in [ImageTransform.rotate_90, ImageTransform.rotate_270, ImageTransform.transpose]:
    return (h, w)
  else:
    return (w, h)


def resize_bilinear(src, size, scale=None, transform:ImageTransform=ImageTransform.none, dtype=None):
  if dtype is None:
    dtype = types.ti_type(src)

  if scale is None:
    scale = tm.vec2(size) / tm.vec2(src.shape[:2])

  size = transformed_size(size, transform)

  dst = types.empty_like(src, (size[1], size[0], 3), dtype)
  f = bilinear_kernel(types.ti_type(src), dtype)
  f(src, dst, tm.vec2(scale), transform)
  return dst

def resize_width(src, width:int, transform:ImageTransform=ImageTransform.none, dtype=None):
  h, w = src.shape[:2]
  scale = width / w
  size = tm.ivec2(width, int(h * scale))
  return resize_bilinear(src, size, scale, transform, dtype)

def scale_bilinear(src, scale, transform:ImageTransform=ImageTransform.none, dtype=None):

  h, w = src.shape[:2]
  size = tm.vec2(w, h) * scale
  return resize_bilinear(src, tm.ivec2(size), scale, transform, dtype)
  



def transform(src, transform:ImageTransform=ImageTransform.none, dtype=None):
  if dtype is None:
    dtype = types.ti_type(src)

  size = src.shape[:2]
  size = transformed_size(size, transform)

  dst = types.empty_like(src, (size[1], size[0], 3), dtype)
  f = bilinear_kernel(types.ti_type(src), dtype)
  f(src, dst, tm.vec2(1.0), transform)
  return dst