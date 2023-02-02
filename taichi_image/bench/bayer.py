import time
import taichi as ti
import numpy as np
import argparse
from taichi_image.bayer import BayerPattern, rgb_to_bayer, bayer_to_rgb

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

  f = bayer_to_rgb_kernel(ti.u8, ti.u8, BayerPattern.RGGB)

  benchmark("bayer_to_rgb_kernel", 
    f, [bayer, out_rgb], 
    iterations=10000, warmup=1000)



  
if __name__ == "__main__":
  main()