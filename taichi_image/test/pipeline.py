from taichi_image.bayer import rgb_to_bayer
from taichi_image.packed import encode12, decode12
from taichi_image.test.arguments import init_with_args

import numpy as np

import cv2


def make_bayer12(image):
  if image.dtype == np.uint8:
    image = image.astype(np.uint16) * 256

  assert image.dtype == np.uint16

  bayer = rgb_to_bayer(image)
  return encode12(bayer.reshape(-1))
      
  



def main():
  args = init_with_args()
  test_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
  bayer12 = make_bayer12(test_image)