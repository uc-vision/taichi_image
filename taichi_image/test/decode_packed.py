import taichi as ti
import numpy as np

import cv2

from taichi_image.test.arguments import init_with_args
from taichi_image import packed, bayer, tonemap
from taichi_image.test.bayer import display_rgb
from taichi_image.camera_isp import Camera32

import torch

def main():
  args = init_with_args()

  isp = Camera32(bayer.BayerPattern.RGGB, resize_width=1024, device='cpu')
  data = np.load(args.image, allow_pickle=True).view(np.uint8)
  data = torch.from_numpy(data)


  image = isp.load_packed12(data)
  rgb8 = isp.tonemap_reinhard([image], gamma=0.6)
  
  if args.show:
    display_rgb("image", rgb8[0].numpy())


  # # for i in range(4):
  # rgb16u = bayer.bayer_to_rgb(cfa, pattern = bayer.BayerPattern.RGGB)
  # rgb8 = tonemap.tonemap_reinhard(rgb16u)



  

if __name__ == "__main__":
  main()