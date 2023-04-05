from taichi_image.test.arguments import init_with_args

import cv2
from taichi_image.test.bayer import display_rgb
from taichi_image.tonemap import tonemap_reinhard


      


def main():
  args = init_with_args()
  test_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  result = tonemap_reinhard(test_image)

  display_rgb("result", result)
  

if __name__ == "__main__":
  main()
