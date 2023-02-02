import taichi as ti
import numpy as np
import argparse
from taichi_image.bayer import BayerPattern, rgb_to_bayer, bayer_to_rgb

import cv2

from taichi_image.test.arguments import init_with_args


def psnr(img1, img2):
  mse = np.mean((img1 - img2) ** 2)
  return 20 * np.log10(255 / np.sqrt(mse))


cv2_to_rgb = dict(
  RGGB = cv2.COLOR_BAYER_BG2RGB,
  GRBG = cv2.COLOR_BAYER_GB2RGB,
  GBRG = cv2.COLOR_BAYER_GR2RGB,
  BGGR = cv2.COLOR_BAYER_RG2RGB,
)

def make_bayer_images(rgb_image):
  return {pattern.name:rgb_to_bayer(rgb_image, pattern) for pattern in BayerPattern }
  
  
def display_rgb(k, rgb_image):
  cv2.imshow(k, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
  cv2.waitKey(0)

def test_rgb_to_bayer(rgb_image):

  bayer_images = make_bayer_images(rgb_image)
  
  for k, bayer_image in bayer_images.items():
    print(f"{k}: {bayer_image.shape} {bayer_image.dtype}")

    converted_rgb = cv2.cvtColor(bayer_image, cv2_to_rgb[k])
    print(f"{k} PSNR: {psnr(rgb_image, converted_rgb):.2f}")

    display_rgb(k, converted_rgb)

def test_bayer_to_rgb(rgb_image):
  bayer_images = make_bayer_images(rgb_image)
  for k, bayer_image in bayer_images.items():
    print(f"{k}: {bayer_image.shape} {bayer_image.dtype}")


    converted_rgb = bayer_to_rgb(bayer_image, BayerPattern[k])
    print(f"{k} PSNR: {psnr(rgb_image, converted_rgb):.2f}")

    display_rgb(k, converted_rgb)
    


def main():
  args = init_with_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  test_image = test_image.astype(np.float32) / 255
  
  # test_rgb_to_bayer(test_image)
  test_bayer_to_rgb(test_image)



if __name__ == "__main__":
  main()