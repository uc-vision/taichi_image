import taichi as ti
from taichi.math import ivec2, vec3

import numpy as np
import argparse

import cv2

from util import flatten, symmetrical, zip_tuple, u8vec3



ti.init(arch=ti.cuda, debug=False)

bayer_filter_upper = [
  # G at R,B locations
  [
            (-2,),
         (0,  4),
    (-2,  4,  8),
  ],

  #  R  G1
  #  G2 B

  # R at G1 and B at G2
  [
            (1,),
       (-2,  0),
    (-2,  8, 10)
  ],


  # B at G1 and R at G2
  [
           (-2,),  
       ( -2, 8),
    ( 1,  0, 10) 
  ],

  # R at B and B at R
  [
          (-3,),
        (4, 0),
    (-3, 0, 12)
  ],

  # Identity 
  [
          (0,),
        (0, 0),
    (0, 0, 16)
  ]
]

diamond = [ 
    (0, 1),
    (-1, 2),
    (-2, 3),
    (-1, 2),
    (0, 1),
]

def diamond_kernel(weights):

  offsets = [ (i-2, x)
    for i, r in enumerate(diamond)
      for x in range(*r)]

  assert len(offsets) == len(weights), f"incorrect weight length {len(offsets)} != {len(weights)}"
  return tuple(zip(offsets, weights))





def make_bayer_kernels():

  g_rb, r_g1, r_g2, rb_br, ident = [
    symmetrical(w) for w in bayer_filter_upper
  ]

  b_g1 = r_g2
  b_g2 = r_g1

  vec_weights = [ 
    zip_tuple(ident, g_rb, rb_br),  # R
    zip_tuple(r_g1, ident, b_g1),     # G1
    zip_tuple(r_g2, ident, b_g2),     # G2
    zip_tuple(rb_br, g_rb, ident),  # B
  ]

  print(len(vec_weights[0]))

  return [diamond_kernel(w) for w in vec_weights]

bayer_kernels = make_bayer_kernels()

bayer_pattern = dict(
  rggb = (0, 1, 1, 2)
)

@ti.kernel
def rgb_to_bayer(image: ti.types.ndarray(u8vec3, ndim=2),
  bayer: ti.types.ndarray(ti.u8, ndim=2), pattern: ti.template()):
  
  p1, p2, p3, p4 = pattern
  for i, j in ti.ndrange(image.shape[0] // 2, image.shape[1] // 2):
    x, y = i * 2, j * 2

    bayer[x, y] = image[x, y][p1]
    bayer[x + 1, y] = image[x + 1, y][p2]
    bayer[x, y + 1] = image[x + 1, y + 1][p3]
    bayer[x + 1, y + 1] = image[x + 1, y + 1][p4]


@ti.func 
def filter_at(image: ti.template(), 
  weights: ti.template(), i:ivec2, divisor: ti.f32):
  
  image_size = ivec2(image.shape)
  c = vec3(0.0)

  for offset, weight in ti.static(weights):
    idx = ti.math.clamp(i + offset, 0, image_size - 1)
    c += ti.cast(image[idx], ti.f32) * vec3(weight)
  
  return ti.cast(ti.math.clamp(c / divisor, 0, 255), ti.u8)


@ti.kernel
def debayer(bayer: ti.types.ndarray(ti.u8, ndim=2), 
  out: ti.types.ndarray(u8vec3, ndim=2), divisor: ti.f32):

  for i, j in ti.ndrange(bayer.shape[0] // 2, bayer.shape[1] // 2):
    x, y = i * 2, j * 2

    out[x, y] = filter_at(bayer, bayer_kernels[0], ivec2(x, y), divisor) 
    out[x + 1, y] = filter_at(bayer, bayer_kernels[2], ivec2(x + 1, y), divisor) 
    
    out[x, y + 1] = filter_at(bayer, bayer_kernels[1], ivec2(x, y + 1), divisor) 
    out[x + 1, y + 1] = filter_at(bayer, bayer_kernels[3], ivec2(x + 1, y + 1), divisor) 
    

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  bayer = np.zeros((test_image.shape[:2]), dtype=np.uint8)
  rgb_to_bayer(test_image, bayer, bayer_pattern["rggb"])

  out = np.zeros_like(test_image)
  debayer(bayer, out, 16.0)



  cv2.imshow("bayer", cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR))
  cv2.imshow("out", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

  cv2.waitKey(0)


main()