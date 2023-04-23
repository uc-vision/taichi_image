from taichi_image.test.bayer import display_rgb
from taichi_image.test.arguments import init_with_args
import numpy as np
import cv2

from taichi_image import bayer, camera_isp, packed
import taichi as ti

def load_test_image(filename, num_cameras, pattern = bayer.BayerPattern.RGGB):
  test_image = cv2.imread(filename)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

  if test_image.dtype == np.uint8:
    test_image = test_image.astype(np.float32) / 255
  elif test_image.dtype == np.uint16:
    test_image = test_image.astype(np.float32) / 65535
  
  num_cameras = 6
  test_images = [ bayer.rgb_to_bayer( (np.clip(test_image, 0, 1) * 65536).astype(np.uint16), pattern=pattern) 
                 for _ in range(num_cameras) ]

  test_images = [packed.encode12(x) for x in test_images]
  return test_images, test_image


def main():
  args = init_with_args()

  test_images, test_image = load_test_image(args.image, 6, pattern = bayer.BayerPattern.RGGB)
  h, w, _ = test_image.shape
                 
  CameraISP = camera_isp.camera_isp(ti.f32)
  isp = CameraISP(bayer.BayerPattern.RGGB, moving_alpha=1.0, resize_width=512)
  
  images = [isp.load_16u(image) for image in test_images]
  outputs = isp.tonemap_reinhard(images, gamma=0.6)

  if args.show:
    display_rgb("test", outputs[0])


if __name__ == '__main__':
  main()