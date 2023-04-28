


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
    self.isp = camera_isp.Camera16(bayer.BayerPattern.RGGB, moving_alpha=0.1, resize_width=3072, transform=ImageTransform.rotate_270)

  def __call__(self, images):
      
    next = [self.isp.load_packed12(image) for image in images]
    out =  self.isp.tonemap_reinhard(next, gamma=0.6)
    return out


def main():
  args = init_with_args()


  test_packed, test_image = load_test_image(args.image,  bayer.BayerPattern.RGGB)
  h, w, _ = test_image.shape

  test_packed = torch.from_numpy(test_packed).to(device='cuda:0')
  test_images = [test_packed.clone() for _ in range(6)]
  processor = Processor()


  for i in tqdm(range(10000)):  
    out = processor(test_images)


  torch.cuda.synchronize()



if __name__ == '__main__':
  with torch.inference_mode():
    main()