from typing import Tuple
import taichi as ti

from taichi_image.bayer import rgb_to_bayer, bayer_to_rgb, BayerPattern
from taichi_image.packed import encode12, decode12
from taichi_image.test.arguments import init_with_args

import numpy as np

import cv2
from taichi_image.test.bayer import display_rgb
from taichi_image.tonemap import tonemap_reinhard



def make_bayer12(image, pattern=BayerPattern.RGGB):
  if image.dtype == np.uint8:
    image = image.astype(np.uint16) * 256

  assert image.dtype == np.uint16

  bayer16 = rgb_to_bayer(image, pattern)
  return encode12(bayer16.reshape(-1), scaled=True)
      
  
def bayer12_pipeline(image_shape:Tuple[int, int],
  bayer12: ti.types.ndarray(ti.u8, ndim=2), pattern=BayerPattern.RGGB):

  bayer16 = decode12(bayer12, dtype=ti.f16, scaled=True).reshape(image_shape)
  rgb16 = bayer_to_rgb(bayer16, pattern)

  return tonemap_reinhard(rgb16)


def trim_image(image):
  if image.shape[0] % 2 == 1:
    image = image[:-1]
  
  if image.shape[1] % 2 == 1:
    image = image[:, :-1]
  
  return image

def main():
  args = init_with_args()
  test_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
  test_image = trim_image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

  bayer12 = make_bayer12(test_image)
  result = bayer12_pipeline(test_image.shape[:2], bayer12)
  display_rgb("result", result)
  

  # import torch

  # result = bayer12_pipeline(test_image.shape[:2], 
  #                           torch.from_numpy(bayer12).to(args.device))

  # display_rgb("result", result.cpu().numpy())

if __name__ == "__main__":
  main()
