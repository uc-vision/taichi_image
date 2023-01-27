import taichi as ti
import numpy as np
import argparse
from taichi_image import BayerPattern, rgb_to_bayer, bayer_to_rgb

import cv2


def psnr(img1, img2):
  mse = np.mean((img1 - img2) ** 2)
  return 20 * np.log10(255 / np.sqrt(mse))


cv2_to_rgb = dict(
  RGGB = cv2.COLOR_BAYER_BG2RGB,
  GRBG = cv2.COLOR_BAYER_GR2RGB,
  GBRG = cv2.COLOR_BAYER_GB2RGB,
  BGGR = cv2.COLOR_BAYER_RG2RGB,
)

def make_bayer_images(rgb_image):
  return {pattern.name:rgb_to_bayer(rgb_image, pattern) for pattern in BayerPattern }
  
  
def test_rgb_to_bayer(rgb_image):

  bayer_images = make_bayer_images(rgb_image)
  for k, bayer_image in bayer_images.items():

    converted_rgb = cv2.cvtColor(bayer_image, cv2_to_rgb[k])
    print(f"{k} PSNR: {psnr(rgb_image, converted_rgb):.2f}")

    cv2.imshow(k, cv2.cvtColor(converted_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

def test_bayer_to_rgb(rgb_image):
  bayer_images = make_bayer_images(rgb_image)
  for k, bayer_image in bayer_images.items():

    converted_rgb = bayer_to_rgb(bayer_image, BayerPattern[k])
    print(f"{k} PSNR: {psnr(rgb_image, converted_rgb):.2f}")

    cv2.imshow(k, cv2.cvtColor(converted_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)



def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  test_bayer_to_rgb(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
  # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  # bayer = np.zeros((test_image.shape[:2]), dtype=np.uint8)
  # rgb_to_bayer(test_image, bayer, BayerPattern.RGGB)

  # out = np.zeros_like(test_image)
  # bayer_to_rgb(bayer, out, 16.0)

  # cv2.imshow("bayer", cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR))
  # cv2.imshow("out", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

  # cv2.waitKey(0)


if __name__ == "__main__":
  main()