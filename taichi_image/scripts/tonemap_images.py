import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import Callable, Iterator, List, Tuple
from natsort import natsorted
import torch
from tqdm import tqdm
from taichi_image.test.arguments import add_taichi_args

from taichi_image import bayer, camera_isp
import taichi as ti

import tifffile


def is_image_file(f:Path, suffixes:List[str]):
  return f.is_file() and f.suffix in suffixes

def find_images(folder:Path, suffixes:List[str]) -> List[str]:
  return natsorted([f.name for f in folder.iterdir() if is_image_file(f, suffixes)])

def strided_image(image, stride = 8):
  return image[::stride, ::stride, :]

@dataclass
class ImageData:
  cpu_image:torch.Tensor
  strided_image:torch.Tensor



def load_image(isp:camera_isp.Camera32, filename:Path, stride = 8) -> ImageData:
  raw = tifffile.imread(filename)

  raw_cuda = raw.to(torch.device('cuda'), non_blocking=True)
  image = isp.load_packed12(raw_cuda)

  cpu_image = torch.empty(image.shape, dtype=torch.uint8, device='cpu', pin_memory=True)
  cpu_image.copy_(image)

  return ImageData(cpu_image, strided_image(image, stride))



def load_images(f : Callable[[Path], ImageData], 
        folder:Path, names:List[str], j:int=os.cpu_count()) -> Iterator[Tuple[str, ImageData]]:
  """ Load a set of folders containing raw images with matching names 
       Returns an iterator of image tuples (camera_id, image_tensor)
  """  
  with ThreadPoolExecutor(max_workers=j) as executor:
    for name, image in executor.map(f, [folder / name for name in names]):
      yield name, image



def main():
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)

  parser = argparse.ArgumentParser()
  parser.add_argument("image_path", type=Path)
  parser.add_argument("--output_path", type=Path)

  parser.add_argument("--width", type=int, default=4096)

  # tonemap parameters
  parser.add_argument("--gamma", type=float, default=0.5)
  parser.add_argument("--intensity", type=float, default=2.0)
  parser.add_argument("--color_adapt", type=float, default=0.2)
  parser.add_argument("--light_adapt", type=float, default=1.0)
  parser.add_argument("--moving_alpha", type=float, default=0.02)
  

  parser.add_argument("--resize_width", type=int, default=0)
  add_taichi_args(parser)

  args = parser.parse_args()

  ti.init(debug=args.debug, 
    arch=ti.cuda if args.device == "cuda" else ti.cpu,  log_level=args.log)

  isp = camera_isp.Camera32(bayer.BayerPattern.RGGB, 
                            transform=args.transform,
                            moving_alpha=args.moving_alpha, 
                            resize_width=args.resize_width)

  assert args.image_path.is_dir(), "image_path must be a directory"
  names = find_images(args.image_path, ['.tiff', '.raw'])

  images_iter = load_images(partial(load_image, isp=isp), args.image_path, names)

  images = list(tqdm(images_iter, total=len(names)))



if __name__ == '__main__':
  main()