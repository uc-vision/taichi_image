import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple
from beartype import beartype
from natsort import natsorted
import torch
from tqdm import tqdm
from taichi_image.interpolate import ImageTransform
from taichi_image.test.bayer import display_rgb
from taichi_image.test.arguments import add_taichi_args

from taichi_image import bayer, camera_isp
import taichi as ti

def is_image_file(f:Path):
  return f.is_file() and f.suffix in ['.tiff', '.raw']

def find_images(folder:Path) -> List[Path]:
  return natsorted([f.name for f in folder.iterdir() if is_image_file(f)])

@beartype
def find_folder_images(folder:Path) -> Tuple[List[Path], List[str]]:
  return [folder], find_images(folder)


@beartype
def set_intersections(image_sets):
  common = set(image_sets[0])
  for images in image_sets[1:]:
    common.intersection_update(set(images))
  return list(common)

@beartype
def find_scan_images(scan_folder:Path) -> Tuple[List[Path], List[str]]:
  cam_folders = {f.name: images for f in scan_folder.iterdir() if f.is_dir() and len(images:=find_images(f)) > 0}
  common_images = set_intersections(list(cam_folders.values()))

  cam_ids = sorted(cam_folders.keys())

  if len(common_images) == 0:
    raise ValueError(f"No common images found in {cam_ids}")

  print(f"Found {cam_ids} image folders with {len(common_images)} images")
  return [scan_folder / id for id in cam_ids], sorted(common_images)

  
def find_scan_folders(scan_folder:Path)  -> Tuple[List[Path], List[str]]:
  folder = Path(scan_folder)
  if not folder.is_dir(): 
    raise FileNotFoundError(f"Folder {folder} does not exist or is not a directory")
    
  return find_scan_images(folder)



@beartype
def load_raw_bytes(filepath, device:torch.device=torch.device('cuda')):
  """Load raw image bytes into torch tensor without any decoding"""
  with open(filepath, 'rb') as f:
    raw_bytes = f.read()
  return torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device, non_blocking=True)

def load_images_iter(f : Callable[[Path], torch.Tensor], folders:list[str], names:List[str],  device:torch.device=torch.device('cuda')):
  """ Load a set of folders containing raw images with matching names 
       Returns an iterator of image tuples {camera_id: raw_bytes_tensor}
  """  

  with ThreadPoolExecutor() as executor:
    def add_group(name):
      return {folder: executor.submit(f, folder / name) 
                for folder in folders}

    group = add_group(names[0])
    for i in range(0, len(names)):
      next_group = add_group(names[i])

      if i > 0: 
        result = {k: future.result() for k, future in group.items()}
        group = next_group
        yield names[i - 1], result





def main():
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)

  parser = argparse.ArgumentParser()
  parser.add_argument("--scan", type=Path)
  parser.add_argument("--images", type=Path)

  parser.add_argument("--width", type=int, default=4096)

  # tonemap parameters
  parser.add_argument("--gamma", type=float, default=0.5)
  parser.add_argument("--intensity", type=float, default=2.0)
  parser.add_argument("--color_adapt", type=float, default=0.2)
  parser.add_argument("--light_adapt", type=float, default=1.0)
  parser.add_argument("--moving_alpha", type=float, default=0.02)

  parser.add_argument("--resize_width", type=int, default=3072)
  parser.add_argument("--transform", type=ImageTransform, default=ImageTransform.rotate_90)
  add_taichi_args(parser)

  args = parser.parse_args()

  ti.init(debug=args.debug, 
    arch=ti.cuda if args.device == "cuda" else ti.cpu,  log_level=args.log)

  isp = camera_isp.Camera32(bayer.BayerPattern.RGGB, 
                            transform=args.transform,
                            moving_alpha=args.moving_alpha, 
                            resize_width=args.resize_width)

  if args.scan is not None:
    folders, names = find_scan_folders(args.scan)
  elif args.images is not None:
    folders, names = find_folder_images(args.images)
  else:
    raise ValueError("No --scan or --images specified")

  images = load_images_iter(partial(load_raw_bytes, device=torch.device(args.device)), folders, names)

  def load_image(bytes):
    assert bytes.shape[0] % 2 == 0, "bytes must have an even number"
    bytes = bytes.view(-1, (args.width * 3)//2)
    return isp.load_packed12(bytes, ids_format=args.ids_format)


  pbar = tqdm(images, total=len(names))
  for name, images in pbar:
    pbar.set_description(name)

    images = [load_image(bytes) for bytes in images.values()]
    outputs = isp.tonemap_reinhard(images, gamma=args.gamma, 
                            intensity=args.intensity, 
                            color_adapt=args.color_adapt, 
                            light_adapt=args.light_adapt)

    display_rgb("tonemapped", torch.concat(outputs, dim=1))



if __name__ == '__main__':
  main()