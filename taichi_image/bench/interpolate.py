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

def to_channels_last(image:torch.Tensor):
  return image.unsqueeze(0).permute(0, 3, 1, 2)

  
def from_channels_last(image:torch.Tensor):
  return image.squeeze(0).permute(1, 2, 0).contiguous()




@torch.compile
def resize_transform(image:torch.Tensor, scale:float):
  image = to_channels_last(image)
  image = torch.nn.functional.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
  image = torch.rot90(image, k=1, dims=[2, 3])
  image = from_channels_last(image)

  return image


def main():
  args = init_with_args(offline_cache=False)

  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)

  args = parser.parse_args()

  test_image = cv2.imread(args.image)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
  
  device = torch.device('cuda', 0)
  test_image = torch.from_numpy(test_image).to(device, dtype=torch.float16) / 255

  benchmark("interpolate", 
    resize_transform, [test_image, 0.8], 
    iterations=10000, warmup=1000)


  benchmark("scale_bilinear", 
    scale_bilinear, [test_image, 0.8, ImageTransform.rotate_90], 
    iterations=10000, warmup=1000)






  
if __name__ == "__main__":
  main()