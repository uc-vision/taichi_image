from pathlib import Path
import taichi as ti
import numpy as np

import cv2
import torch

from taichi_image.test.arguments import init_with_args
from taichi_image import packed, bayer, tonemap
from taichi_image.test.bayer import display_rgb, load_rgb


def main():
  args = init_with_args()

  filename = Path(args.image)
  
  if filename.suffix == ".npy":
    data = np.load(args.image, allow_pickle=True).view(np.uint8)

  if filename.suffix == ".pt":
    data = torch.load(args.image).cpu().numpy().view(np.uint8)

  elif filename.suffix in [".png", ".jpg", ".jpeg"]:
    image = load_rgb(filename).astype(np.uint16) * 16
    cfa = bayer.rgb_to_bayer(image, pattern = bayer.BayerPattern.GBRG)

    data = packed.encode12(cfa, ids_format=args.ids_format)


  decoded_cfa = packed.decode12(data, scaled=True, ids_format=args.ids_format)
  rgb = bayer.bayer_to_rgb(decoded_cfa, pattern = bayer.BayerPattern.GBRG)

  print(rgb.shape)

  rgb = tonemap.tonemap_reinhard(rgb)

  

  if args.show:
    display_rgb("image", rgb)




  

if __name__ == "__main__":
  main()