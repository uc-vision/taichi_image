


import torch
from taichi_image.bench.util import benchmark
from taichi_image.test.bayer import display_rgb
from test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp, packed
import taichi as ti

from concurrent.futures import ThreadPoolExecutor



def main():
  args = init_with_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  if test_image.dtype == np.uint8:
    test_image = test_image.astype(np.float32) / 255
  elif test_image.dtype == np.uint16:
    test_image = test_image.astype(np.float32) / 65535

  pattern = bayer.BayerPattern.RGGB
  num_cameras = 6
  test_images = [ bayer.rgb_to_bayer( (np.clip(test_image, 0, 1)).astype(np.uint16), pattern=pattern) 
                 for _ in range(num_cameras)]

  test_images = [packed.encode12(x.reshape(-1)) for x in test_images]
  h, w, _ = test_image.shape

  test_images = [torch.from_numpy(x).to(device='cuda:0') for x in test_images]
                 
  CameraISP = camera_isp.camera_isp(ti.f16)
  isp = CameraISP(pattern, moving_alpha=0.1, resize_width=2560)


  def f():
    images = [isp.load_packed12(image, image_size=(w, h)) for image in test_images]
    isp.tonemap_reinhard(images, gamma=0.6)


  benchmark("camera_isp", 
    f, [], 
    iterations=1000, warmup=100)   
   


if __name__ == '__main__':
  main()