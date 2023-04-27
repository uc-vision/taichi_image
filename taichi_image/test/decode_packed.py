from pathlib import Path
import taichi as ti
import numpy as np

import cv2

from taichi_image.test.arguments import init_with_args
from taichi_image import packed, bayer, tonemap
from taichi_image.test.bayer import display_rgb, load_rgb


def main():
  args = init_with_args()

  filename = Path(args.image)
  
  if filename.suffix == ".npy":
    data = np.load(args.image, allow_pickle=True).view(np.uint8)

  elif filename.suffix == ".jpg":
    image = load_rgb(filename).astype(np.uint16) * 16
    cfa = bayer.rgb_to_bayer(image, pattern = bayer.BayerPattern.RGGB)

    data = packed.encode12(cfa)

  decoded_cfa = packed.decode12(data, scaled=True)
  rgb = bayer.bayer_to_rgb(decoded_cfa, pattern = bayer.BayerPattern.RGGB)

  rgb = tonemap.tonemap_linear(rgb)

  if args.show:
    display_rgb("image", rgb)




  

if __name__ == "__main__":
  main()