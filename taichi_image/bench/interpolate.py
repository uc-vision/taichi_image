import time
import taichi as ti
import numpy as np
import argparse
from taichi_image.interpolate import  scale_bilinear, ImageTransform, transform

import cv2
import torch

import tqdm as tqdm

from taichi_image.test.arguments import init_with_args
from taichi_image.bench import benchmark

import torch.nn.functional as F


def resize_transform(image:torch.Tensor, scale:float):
  
  image = scale_bilinear(image, scale)
  return torch.rot90(image, k=1, dims=[0, 1]).contiguous()

@torch.compile
def interpolate_transform(image:torch.Tensor, scale:float):
  image = image.permute(2, 0, 1).unsqueeze(0)
  image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
  image = image.squeeze(0).permute(1, 2, 0)
  return torch.rot90(image, k=1, dims=[0, 1]).contiguous()


def main():
  args = init_with_args(offline_cache=False)

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
  
  device = torch.device('cuda', 0)
  test_image = torch.from_numpy(test_image).to(device, dtype=torch.float16) / 255

  # benchmark("interpolate_transform", 
  #   interpolate_transform, [test_image, 0.8], 
  #   iterations=10000, warmup=1000)
  benchmark("scale_bilinear", 
    scale_bilinear, [test_image, 0.8], 
    iterations=10000, warmup=1000)

  benchmark("resize_transform", 
    resize_transform, [test_image, 0.8], 
    iterations=10000, warmup=1000)









  
if __name__ == "__main__":
  main()