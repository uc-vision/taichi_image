


import torch
from taichi_image.bench.util import benchmark
from taichi_image.test.bayer import display_rgb
from taichi_image.test.camera_isp import load_test_image
from test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp, packed
import taichi as ti

from concurrent.futures import ThreadPoolExecutor



def main():
  args = init_with_args()

  test_images, test_image = load_test_image(args.image, 6, pattern = bayer.BayerPattern.RGGB)
  h, w, _ = test_image.shape

  test_images = [torch.from_numpy(x).to(device='cuda:0') for x in test_images]
                 
  CameraISP = camera_isp.camera_isp(ti.f16)
  isp = CameraISP(bayer.BayerPattern.RGGB, moving_alpha=0.1, resize_width=2560)


  def f():
    next = [isp.load_packed12(image, image_size=(w, h)) for image in test_images]
    isp.tonemap_reinhard(next, gamma=0.6)


  benchmark("camera_isp", 
    f, [], 
    iterations=1000, warmup=100)   
   


if __name__ == '__main__':
  main()