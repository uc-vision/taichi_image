import torch
from taichi_image.test.bayer import display_rgb
from taichi_image.test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp, packed
import taichi as ti

def load_test_image(filename, pattern = bayer.BayerPattern.RGGB):
  test_image = cv2.imread(filename)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  if test_image.dtype == np.uint8:
    test_image = test_image.astype(np.float32) / 255
  elif test_image.dtype == np.uint16:
    test_image = test_image.astype(np.float32) / 65535
  
  cfa =  bayer.rgb_to_bayer( (np.clip(test_image, 0, 1) * 65536).astype(np.uint16), pattern=pattern) 
  raw = packed.encode12(cfa) 
  return raw, test_image


def load_test_images(filename, num_cameras, pattern = bayer.BayerPattern.RGGB):
  raw_image, test_image = load_test_image(filename, pattern)
  return [raw_image] * num_cameras, test_image


def main():
  args = init_with_args()

  raw_image, test_image = load_test_image(args.image, pattern = bayer.BayerPattern.RGGB)
  isp = camera_isp.Camera32(bayer.BayerPattern.RGGB, moving_alpha=1.0, resize_width=512)

  
  # image = isp.load_packed12(torch.from_numpy(raw_image)) 
  outputs = isp.tonemap_reinhard(torch.from_numpy(test_image).to(torch.float32), gamma=0.6)



  if args.show:
    display_rgb("test", outputs)


if __name__ == '__main__':
  main()