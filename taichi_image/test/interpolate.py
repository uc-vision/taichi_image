import taichi as ti
import numpy as np

import cv2

from taichi_image.test.arguments import init_with_args
from taichi_image.interpolate import resize_bilinear, scale_bilinear

def test_resize(test_image):
  # out =  resize_bilinear(test_image, (1024, 1024))
  out = scale_bilinear(test_image, 0.125)
  cv2.imshow("out", out)
  cv2.waitKey(0)


def main():
  args = init_with_args()

  test_image = cv2.imread(args.image)
  # test_image = test_image.astype(np.float32) / 255
  
  # test_rgb_to_bayer(test_image)
  test_resize(test_image)



if __name__ == "__main__":
  main()