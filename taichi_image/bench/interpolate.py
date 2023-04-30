import time
import taichi as ti
import numpy as np
import argparse
from taichi_image.interpolate import resize_bilinear, scale_bilinear

import cv2
import torch

import tqdm as tqdm

from taichi_image.test.arguments import init_with_args
from taichi_image.bench import benchmark


def main():
  args = init_with_args(offline_cache=False)

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  
  device = torch.device('cuda', 0)
  bayer = torch.from_numpy(bayer).to(device)

  benchmark("bayer_to_rgb_kernel", 
    scale_bilinear, [test_image, 0.5], 
    iterations=10000, warmup=1000)


  
if __name__ == "__main__":
  main()