import enum
import taichi as ti
from taichi.math import ivec2, vec3

import numpy as np

from taichi_image.kernel import symmetrical, zip_tuple, u8vec3



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
  return filter_kernel(tuple(zip(offsets, weights)))


def filter_kernel(kernel):
  return tuple((i, w) for i, w in kernel if w != 0)

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


bayer_kernels_5x5 = make_bayer_kernels()



class BayerPattern(enum.Enum):
  RGGB = 0
  GRBG = 1
  GBRG = 2
  BGGR = 3

  @property
  def pixel_order(self):
    return pixel_orders[self.value]

pixel_orders = {
    BayerPattern.RGGB.value : (0, 1, 1, 2),
    BayerPattern.GRBG.value : (1, 0, 2, 1),
    BayerPattern.GBRG.value : (1, 2, 0, 1),
    BayerPattern.BGGR.value : (2, 1, 1, 0)
}
    


@ti.kernel
def rgb_to_bayer_kernel(image: ti.types.ndarray(u8vec3, ndim=2),
                 bayer: ti.types.ndarray(ti.u8,
                                         ndim=2), pixel_order: ti.template()):

  p1, p2, p3, p4 = pixel_order
  for i, j in ti.ndrange(image.shape[0] // 2, image.shape[1] // 2):
    x, y = i * 2, j * 2

    bayer[x, y] = image[x, y][p1]
    bayer[x + 1, y] = image[x + 1, y][p2]
    bayer[x, y + 1] = image[x + 1, y + 1][p3]
    bayer[x + 1, y + 1] = image[x + 1, y + 1][p4]

def scale_factor(dtype):
  if dtype == ti.u8:
    return 255
  elif dtype == ti.u16:
    return 65535
  elif dtype == ti.f32:
    return 1.0
  else:
    raise ValueError(f"unsupported dtype {dtype}")


# @ti.func
# def read_pixel(image:ti.template(), i:ivec2):
#   return ti.cast(image[i], ti.f32) / ti.static(scale_factor(image.dtype))

@ti.func
def filter_at(
    image: ti.template(), weights: ti.template(), i: ivec2):

  image_size = ivec2(image.shape)
  c = vec3(0.0)

  for offset, weight in ti.static(weights):
    idx = ti.math.clamp(i + offset, 0, image_size - 1)
    c += ti.cast(image[idx], ti.f32) * vec3(weight)

  return ti.cast(ti.math.clamp(c / 16, 0, 255), ti.u8)

@ti.func
def debayer_at(bayer:ti.template(), i:ivec2,
     kernels:ti.template()) -> u8vec3:
  offset = i % 2 
  idx = offset.x + offset.y * 2


  c = u8vec3(0)
  if idx == 0:
    c =  filter_at(bayer, kernels[0], i)
  elif idx == 1:
    c =  filter_at(bayer, kernels[1], i)
  elif idx == 2:
    c =  filter_at(bayer, kernels[2], i)
  elif idx == 3:
    c =  filter_at(bayer, kernels[3], i)
  
  return c



@ti.func
def bayer_2x2(bayer: ti.template(), out: ti.template(), kernels: ti.template(),
              i: ti.i32, j: ti.i32):

    out[i, j] = filter_at(bayer, kernels[0], ivec2(i, j))
    out[i + 1, j] = filter_at(bayer, kernels[1], ivec2(i + 1, j))

    out[i, j + 1] = filter_at(bayer, kernels[2], ivec2(i, j + 1))
    out[i + 1, j + 1] = filter_at(bayer, kernels[3], ivec2(i + 1, j + 1))


@ti.kernel
def bayer_to_rgb_kernel(bayer: ti.types.ndarray(ti.u8, ndim=2),
            out: ti.types.ndarray(u8vec3, ndim=2), kernels: ti.template()):

  for i, j in ti.ndrange(bayer.shape[0] // 2, bayer.shape[1] // 2):
    bayer_2x2(bayer, out, kernels, i * 2, j * 2)



def rgb_to_bayer(image, pattern:BayerPattern):
  assert image.ndim == 3 and image.shape[2] == 3, "image must be RGB"

  bayer = np.empty(image.shape[:2], dtype=np.uint8)
  rgb_to_bayer_kernel(image, bayer, pattern.pixel_order)
  return bayer



kernel_patterns = {
    BayerPattern.RGGB: [0, 1, 2, 3],
    BayerPattern.GRBG: [1, 0, 3, 2],
    BayerPattern.GBRG: [2, 3, 0, 1],
    BayerPattern.BGGR: [3, 2, 1, 0],
}
  
def bayer_kernels(pattern:BayerPattern):
  return tuple([bayer_kernels_5x5[p] for p in kernel_patterns[pattern]])

def bayer_to_rgb(bayer, pattern:BayerPattern):
  assert bayer.ndim == 2 , "image must be mono bayer"


  rgb = np.empty((*bayer.shape, 3), dtype=np.uint8)
  bayer_to_rgb_kernel(bayer, rgb, bayer_kernels(pattern))
  return rgb