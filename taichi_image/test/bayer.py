import taichi as ti
import numpy as np
import argparse
from taichi_image.bayer import BayerPattern, rgb_to_bayer, bayer_to_rgb
from taichi_image.util import cache
from typing import List, Tuple

import cv2
import torch

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
  

def load_rgb(filename):
  image = cv2.imread(str(filename))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image
  
def display_rgb(k, rgb_image):
  if isinstance(rgb_image, torch.Tensor):
    rgb_image = rgb_image.cpu().numpy()
    
  cv2.namedWindow(k, cv2.WINDOW_NORMAL)

  key = -1
  while key == -1:
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.putText(img, 'Hello world!', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 255, 255), 10, 2)
    cv2.imshow(k, img)
    key = cv2.waitKey(1)


def display_multi_rgb(window_name: str, rgbs: List[Tuple[np.ndarray, str]], continuous: bool = False):
  """ 
    Displays multiple rgb images provided in a list of 
    tuple where the first item is the image and second a caption of parameters for the image.
    continuous - Bool flag whether to wait for key before returning result.
  """
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window_name, 2304, 1296) 
  images = []
  for image, text in rgbs:
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(img, text, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (50, 50, 255), 16, 4)
    images.append(img)
  window_image = np.concatenate(images, axis=0)
  
  if continuous:
    cv2.imshow(window_name, window_image)
    key = cv2.waitKey(10)
  else:
    key = -1
    while key == -1:
      cv2.imshow(window_name, np.concatenate(images, axis=0))
      key = cv2.waitKey(10)
  return key == 13  # Returns if enter was pressed.



def save_rgb(k, rgb_image):
  if isinstance(rgb_image, torch.Tensor):
    rgb_image = rgb_image.cpu().numpy()

  key = -1
  while key == -1:
    resized_image = cv2.resize(rgb_image, (0, 0,), fx=0.3, fy=0.3)
    cv2.imwrite('/local/kla129/taichi_image/' + k + '.png', cv2.cvtColor(resized_image, cv2.COLOR_RGBA2GRAY))  # COLOR_RGB2BGR
    # key = input('enter key to continue') 
    key = 2

@cache
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