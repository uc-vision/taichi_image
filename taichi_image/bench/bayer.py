import time
import taichi as ti
import numpy as np
import argparse
from taichi_image import BayerPattern, rgb_to_bayer, bayer_to_rgb
from taichi_image.bayer import u8vec3

import cv2
import torch

import tqdm as tqdm

from taichi_image.bayer import bayer_kernels, bayer_to_rgb_kernel
from taichi_image.test.arguments import init_with_args

from taichi_image.bench import benchmark


def main():
  args = init_with_args(offline_cache=False)

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  bayer = rgb_to_bayer(test_image, BayerPattern.RGGB)

  device = torch.device('cuda', 0)
  bayer = torch.from_numpy(bayer).to(device)

  out_rgb = torch.zeros( (*bayer.shape, 3), dtype=torch.uint8, device=device)
  kernels = bayer_kernels(BayerPattern.RGGB)

  benchmark("bayer_to_rgb_kernel", 
    bayer_to_rgb_kernel, [bayer, out_rgb, kernels], 
    iterations=10000, warmup=1000)



  
if __name__ == "__main__":
  main()