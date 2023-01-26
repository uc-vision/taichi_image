import taichi as ti
import numpy as np
import argparse

import cv2


ti.init(arch=ti.cuda, debug=True)

cross = [
          (-1,),
       (0,  2),
  (-1,  2,  4),
  ]

uniform = [
  (1,  1,  1),
  (1,  -8,  1),
  (1,  1,  1),
  ]



diamond = [ 
    (0, 1),
    (-1, 2),
    (-2, 3),
    (-1, 2),
    (0, 1),
]


def mirror(w):
  return w + w[:-1][::-1]


def symmetrical(w):
  w = mirror([mirror(row) for row in w])
  return flatten(w)

def flatten(w):
  return [x for row in w for x in row]

def diamond_kernel(weights):

  offsets = [ (i-2, x)
    for i, r in enumerate(diamond)
      for x in range(*r)]

  assert len(offsets) == len(weights), f"incorrect weight length {len(offsets)} != {len(weights)}"
  return tuple(zip(offsets, weights))

def kernel_square(weights, n=5):

  offsets = [ (i, j) for i in range(-(n//2), n//2 + 1) 
    for j in range(-(n//2), n//2 + 1)]

  assert len(offsets) == len(weights), f"incorrect weight length {len(offsets)} != {len(weights)}"
  return tuple(zip(offsets, weights))



u8vec3 = ti.types.vector(3, ti.u8)

@ti.kernel
def norm_convolve(image: ti.types.ndarray(u8vec3, ndim=2), weights: ti.template(), 
  out: ti.types.ndarray(u8vec3, ndim=2)):
  total = ti.static(sum([w for _, w in weights]))

  image_size = ti.math.ivec2(image.shape)
  for i in ti.grouped(image):
    c = ti.math.vec3(0.0)
    for offset, weight in ti.static(weights):
      idx = ti.math.clamp(i + offset, 0, image_size - 1)

      c += ti.cast(image[idx], ti.f32) * weight
    out[i] = ti.cast(ti.math.clamp(c / total, 0, 255), ti.u8)
    


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  out = np.zeros_like(test_image)

  kernel = diamond_kernel(symmetrical(cross))

  # kernel = kernel_square(flatten(uniform), n=3)

  norm_convolve(test_image, kernel, out)


  cv2.imshow("out", out)
  cv2.waitKey(0)


main()