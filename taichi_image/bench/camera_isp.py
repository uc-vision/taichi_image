


import argparse
from functools import partial
import torch
from taichi_image.bench.util import benchmark
from taichi_image.test.bayer import display_rgb
from taichi_image.test.camera_isp import load_test_image
from taichi_image.interpolate import ImageTransform
from tqdm import tqdm
from taichi_image.test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp, packed
import taichi as ti

# from torch.multiprocessing import Pool


class Processor:
  def __init__(self):
    self.isp = camera_isp.Camera16(bayer.BayerPattern.RGGB, moving_alpha=0.1, resize_width=64)

  def __call__(self, image):
      
    next = self.isp.load_packed12(image)
    out =  self.isp.tonemap_reinhard(next, gamma=0.6)
    return out


def main():
  # parser = argparse.ArgumentParser()
  # parser.add_argument("image", type=str)
  # add_taichi_args(parser)
  # args = parser.parse_args()
  args = init_with_args()


  # pool = Pool(1, initializer=partial(ti.init, arch=ti.cuda, device_memory_GB=0.5))
  # test_images, test_image = pool.apply(load_test_image, args=[args.image, 6, bayer.BayerPattern.RGGB])

  test_packed, test_image = load_test_image(args.image,  bayer.BayerPattern.RGGB)
  h, w, _ = test_image.shape

  test_packed = torch.from_numpy(test_packed).to(device='cuda:0')
  processors = [Processor() for i in range(6)]

  for i in tqdm(range(10000)):  
    out = [p(test_packed) for p in processors]


  torch.cuda.synchronize()
  # benchmark("camera_isp", 
  #   f, [test_packed], 
  #   iterations=1000, warmup=100)   
  
  




if __name__ == '__main__':
  with torch.inference_mode():
    main()