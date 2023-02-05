import enum
import taichi as ti
from taichi.math import ivec2, vec3
import torch

import numpy as np

from taichi_image.kernel import flatten, symmetrical, zip_tuple, u8vec3
from taichi_image.util import cache

from . import types

from typeguard import typechecked



def diamond_kernel(weights):
  diamond = [
      (0, 1),
      (-1, 2),
      (-2, 3),
      (-1, 2),
      (0, 1),
  ]
  offsets = [(i - 2, x) for i, r in enumerate(diamond) for x in range(*r)]

  assert len(offsets) == len(
      weights), f"incorrect weight length {len(offsets)} != {len(weights)}"
  return tuple(zip(offsets, weights))


def make_bayer_kernels():
  #  R  G1
  #  G2 B

  g_rb, r_g1, r_g2, rb_br, ident = [
      symmetrical(w) for w in [

          [(-2,), (0, 4), (-2, 4, 8)],  # G at R,B locations
          [(-2,), (-2, 8), (1, 0, 10)], # R at G1 and B at G2
          [(1,), (-2, 0), (-2, 8, 10)], # B at G1 and R at G2
          [(-3,), (4, 0), (-3, 0, 12)], # R at B and B at R
          [(0,), (0, 0), (0, 0, 16)] # Identity (R at R, G at G etc.)
      ]
  ]

  b_g1 = r_g2
  b_g2 = r_g1

  vec_weights = [
      zip_tuple(ident, g_rb, rb_br),  # R
      zip_tuple(r_g1, ident, b_g1),  # G1
      zip_tuple(r_g2, ident, b_g2),  # G2
      zip_tuple(rb_br, g_rb, ident),  # B
  ]

  return tuple([diamond_kernel(w) for w in vec_weights])


bayer_kernels = make_bayer_kernels()


class BayerPattern(enum.Enum):
  RGGB = 0
  GRBG = 1
  GBRG = 2
  BGGR = 3

  @property
  def pixel_order(self):
    return pixel_orders[self]

pixel_orders = {
    BayerPattern.RGGB : (0, 1, 1, 2),
    BayerPattern.GRBG : (1, 0, 2, 1),
    BayerPattern.GBRG : (1, 2, 0, 1),
    BayerPattern.BGGR : (2, 1, 1, 0)
}

kernel_patterns = {
    BayerPattern.RGGB: (0, 1, 2, 3),
    BayerPattern.GBRG: (1, 0, 3, 2),
    BayerPattern.GRBG: (2, 3, 0, 1),
    BayerPattern.BGGR: (3, 2, 1, 0),
}



@ti.kernel
def rgb_to_bayer_kernel(image: ti.types.ndarray(ndim=3),
                bayer: ti.types.ndarray(ndim=2), pixel_order: ti.template()):

  p1, p2, p3, p4 = pixel_order
  for i, j in ti.ndrange(bayer.shape[1] // 2, bayer.shape[0] // 2):
    x, y = i * 2, j * 2

    bayer[y, x] = image[y, x, p1]
    bayer[y, x + 1] = image[y, x + 1, p2]
    bayer[y + 1, x] = image[y + 1, x, p3]
    bayer[y + 1, x + 1] = image[y + 1, x + 1, p4]

@cache    
@typechecked
def bayer_to_rgb_kernel(pattern:BayerPattern, in_dtype=ti.u8, out_dtype=None):
  if out_dtype is None:
    out_dtype = in_dtype

  in_scale = types.pixel_types[in_dtype]
  out_scale = types.pixel_types[out_dtype]
  
  out_vec3 = ti.types.vector(3, out_dtype)
  kernels =  tuple([bayer_kernels[i] for i in kernel_patterns[pattern] ])

  @ti.func
  def write_pixel(image: ti.template(), i: ivec2, v: vec3):
    image[i] = ti.cast(v * out_scale, out_dtype)


  @ti.func
  def filter_at(
      image: ti.template(), weights: ti.template(), i: ivec2) -> vec3:

    image_size = ivec2(image.shape)
    c = vec3(0.0)

    for offset, weight in ti.static(weights):
      idx = ti.math.clamp(i + offset, 0, image_size - 1)
      c += ti.cast(image[idx], ti.f32) * vec3(weight)

    return ti.math.clamp(c / (in_scale * 16.0), 0, 1.0)


  @ti.kernel
  def f(bayer: ti.types.ndarray(dtype=in_dtype, ndim=2),
              out: ti.types.ndarray(dtype=out_vec3, ndim=2)):

    for i, j in ti.ndrange(bayer.shape[0] // 2, bayer.shape[1] // 2):
      x, y = i * 2, j * 2

      write_pixel(out, ivec2(x, y), 
        filter_at(bayer, kernels[0], ivec2(x, y)))
        
      write_pixel(out, ivec2(x + 1, y), 
        filter_at(bayer, kernels[1], ivec2(x + 1, y)))

      write_pixel(out, ivec2(x, y + 1), 
        filter_at(bayer, kernels[2], ivec2(x, y + 1)))
    
      write_pixel(out, ivec2(x + 1, y + 1),
        filter_at(bayer, kernels[3], ivec2(x + 1, y + 1)))


  return f


def rgb_to_bayer(image, pattern:BayerPattern):
  assert image.ndim == 3 and image.shape[2] == 3, "image must be RGB"

  bayer = types.zeros_array(image, image.shape[:2], dtype=image.dtype)
  rgb_to_bayer_kernel(image, bayer, pattern.pixel_order)
  return bayer


  
def bayer_to_rgb(bayer, pattern:BayerPattern, dtype=None):
  assert bayer.ndim == 2 , "image must be mono bayer"

  
  rgb = types.empty_array(bayer, shape=bayer.shape + (3,), dtype=dtype)
  f = bayer_to_rgb_kernel(pattern, types.ti_type(bayer), types.ti_type(rgb))

  f(bayer, rgb)
  return rgb



    

