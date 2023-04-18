


import torch
from taichi_image.bench.util import benchmark
from taichi_image.test.bayer import display_rgb
from test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp
import taichi as ti



def main():
  args = init_with_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  if test_image.dtype == np.uint8:
    test_image = test_image.astype(np.float32) / 255
  elif test_image.dtype == np.uint16:
    test_image = test_image.astype(np.float32) / 65535

  pattern = bayer.BayerPattern.RGGB
  test_images = [ bayer.rgb_to_bayer( (np.clip(test_image * x, 0, 1)).astype(np.float16), pattern=pattern) for x in [0.2, 0.4, 0.8]]
  image_size = (test_image.shape[1], test_image.shape[0]) 

  test_images = [torch.from_numpy(x).to(device='cuda:0') for x in test_images]
                 
  CameraISP = camera_isp.camera_isp(ti.f32)

  isp = CameraISP(len(test_images), image_size, pattern, moving_alpha=1.0, resize_width=512)

  def f():
    isp.load_16f(test_images)
    outputs = isp.outputs_like(test_images)
    isp.tonemap_reinhard(outputs, gamma=0.6)


  benchmark("camera_isp", 
    f, [], 
    iterations=1000, warmup=100)   
   


if __name__ == '__main__':
  main()