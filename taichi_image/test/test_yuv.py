import argparse 
import cv2
import numpy as np
import torch

from taichi_image import color_conversion
from taichi_image.test.arguments import init_with_args


def show_yuv420(img):
  if isinstance(img, torch.Tensor):
    img = img.cpu().numpy()

  bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)

  cv2.imshow('img', bgr)
  cv2.waitKey(0)


def main():
    args = init_with_args()

    img = cv2.imread(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    yuv2 = color_conversion.rgb_yuv420_image(rgb)


    if args.show:
      show_yuv420(yuv)
      show_yuv420(yuv2)
        


        


if __name__ == '__main__':
    main()
