import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple
from beartype import beartype
from natsort import natsorted
import torch
import numpy as np
from tqdm import tqdm
from taichi_image.interpolate import ImageTransform
from taichi_image.test.bayer import display_rgb, save_rgb, display_multi_rgb
from taichi_image.test.arguments import add_taichi_args

from time import perf_counter

from taichi_image import bayer, monodepth_camera_isp, camera_isp
import taichi as ti


from monodepth_utilities.algorithms import CachedDepthAnythingV2TRTPredictor
from monodepth_utilities.utils.imports import set_depth_anything_v2_dir


class TonemapParams:

  def __init__(self, gamma: float, intensity: float, colour_adapt: float, light_adapt: float, width = 4096, ids_format=None, isp=None):
    self.gamma = gamma
    self.intensity = intensity
    self.colour_adapt = colour_adapt
    self.light_adapt = light_adapt
    self.width = width
    self.ids_format = ids_format
    self.isp = isp

  @property
  def kwargs(self) -> dict:
    return dict(gamma=self.gamma, intensity=self.intensity, color_adapt=self.colour_adapt, light_adapt=self.light_adapt)
  
  @property
  def args(self) -> list:
    return self.gamma, self.intensity, self.colour_adapt, self.light_adapt
  
  @property
  def isp_name(self) -> str:
    if self.isp is None:
      return 'None'
    return 'Depth' if 'monodepth' in str(self.isp.__class__).lower() else 'Stride'

  def to_rgb(self, outputs: torch.Tensor) -> Tuple[np.ndarray, str]:
    """ Returns a tuple of the image converted to a numpy array and a string representing the parameters. """
    return (
      np.concatenate([o.cpu().numpy() for o in outputs], axis=1),
      str(self)
    )
  
  def __str__(self) -> str:
    return f'Mode: {self.isp_name}, Gamma: {self.gamma}, Intensity: {self.intensity}, Colour Adapt: {self.colour_adapt}, Light Adapt: {self.light_adapt}'
  

DEFAULT_PARAMS = TonemapParams(0.5, 2.0, 0.2, 1.0)


def get_default(array, index, default_value):
  try:
    return array[index]
  except IndexError:
    return default_value


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


def load_image(bytes, param: TonemapParams):
  """ Load image bytes to torch tensor"""
  assert bytes.shape[0] % 2 == 0, "bytes must have an even number"
  bytes = bytes.view(-1, (param.width * 3)//2)
  return param.isp.load_packed12(bytes, ids_format=param.ids_format)


def tonemap_scan_parser():
  """ Create and setup the argument parser for the tonemap scan. """
  parser = argparse.ArgumentParser()
  parser.add_argument("--scan", type=Path, default='/uc/research/CropVision/raw/scan_10-52-18')
  parser.add_argument("--images", type=Path)

  parser.add_argument("--width", type=int, default=4096)

  parser.add_argument("--gamma", nargs='*', type=float, default=[])
  parser.add_argument("--intensity", nargs='*', type=float, default=[])
  parser.add_argument("--color_adapt", nargs='*', type=float, default=[])  # was 0.2
  parser.add_argument("--light_adapt", nargs='*', type=float, default=[])
  # parser.add_argument("--moving_alpha", nargs='*', type=float)

  parser.add_argument("--resize_width", type=int, default=3072)
  parser.add_argument("--transform", type=ImageTransform, default=ImageTransform.rotate_90)
  parser.add_argument('--depth_camera', action='store_true')
  parser.add_argument("--stride_camera", action='store_true')
  parser.add_argument("--reverse", action='store_true')
  parser.add_argument("--depth_debug", action='store_true')
  parser.add_argument("--dav2_module", type=Path)
  add_taichi_args(parser)

  return parser


def parse_tonemap_params(args) -> List[TonemapParams]:
  """ Parse arguments into a list of tonemap parameters. """
  params = []
  for i in range(max(len(args.gamma), len(args.intensity), len(args.color_adapt), len(args.light_adapt))):
    params.append(TonemapParams(
      get_default(args.gamma, i, DEFAULT_PARAMS.gamma),
      get_default(args.intensity, i, DEFAULT_PARAMS.intensity),
      get_default(args.color_adapt, i, DEFAULT_PARAMS.colour_adapt),
      get_default(args.light_adapt, i, DEFAULT_PARAMS.light_adapt),
      args.width,
      args.ids_format,
      monodepth_camera_isp.MonoDepthCamera32(
        bayer.BayerPattern.RGGB, 
        transform=args.transform,
        moving_alpha=0.02, 
        resize_width=args.resize_width,
        stride=8,
        debug=args.depth_debug,
        predictor=CachedDepthAnythingV2TRTPredictor()
    )))
  if args.depth_camera or len(params) == 0:
    # Uses default params for the first one or two cameras depending if stride is active.
    params.insert(0, TonemapParams(*DEFAULT_PARAMS.args, 
      args.width,
      args.ids_format,
      monodepth_camera_isp.MonoDepthCamera32(
        bayer.BayerPattern.RGGB,
        transform=args.transform,
        moving_alpha=0.02,
        resize_width=args.resize_width,
        stride=8,
        debug=args.depth_debug,
        predictor=CachedDepthAnythingV2TRTPredictor()
      )))
  if args.stride_camera:
    # Adds a stride camera to the parameters.
    params.insert(0, TonemapParams(*DEFAULT_PARAMS.args,
      args.width,
      args.ids_format,
      camera_isp.Camera32(
        bayer.BayerPattern.RGGB, 
        transform=args.transform,
        moving_alpha=0.02, 
        resize_width=args.resize_width
    )))
  return params


def print_params(params: List[TonemapParams]):
  print('Running tonemap scan with parameters: ')
  print('|   Mode   |   Gamma   |   Intensity   |   Colour Adapt   |   Light Adapt   |')
  for param in params:
    print('| {:^8} | {:^9} | {:^13} | {:^16} | {:^15} |'.format(param.isp_name, *param.args))
  

def main():
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
  
  args = tonemap_scan_parser().parse_args()

  if args.dav2_module:
    set_depth_anything_v2_dir(args.dav2_module)

  params = parse_tonemap_params(args)
    
  if args.scan is not None:
    folders, names = find_scan_folders(args.scan)
  elif args.images is not None:
    folders, names = find_folder_images(args.images)
  else:
    raise ValueError("No --scan or --images specified")

  if args.reverse:
    names.reverse()

  images = load_images_iter(partial(load_raw_bytes, device=torch.device(args.device)), folders, names)

  ti.init(debug=args.debug, arch=ti.cuda if args.device == "cuda" else ti.cpu,  log_level=args.log)

  try:
    # Create cameras for multiple params.
    print_params(params)
    pbar = tqdm(images, total=len(names))
    for name, images in pbar:
      pbar.set_description(name)
      rgbs = []
      start = perf_counter()
      for param in params:
        outputs = param.isp.tonemap_reinhard([load_image(bytes, param) for bytes in images.values()], **param.kwargs)
        rgbs.append(param.to_rgb(outputs))
      print('?????', len(rgbs), rgbs)
      print(f'took to display: {perf_counter() - start} (s)')

      display_multi_rgb('Tonemapped', rgbs)
      # if depth_flag:
      #   if isp.has_masks():
      #     results.append(np.concatenate(isp.get_transformed_masks(), axis=1))
      #   if isp.has_visuals():
      #     results.append(np.concatenate(isp.get_transformed_visuals(), axis=1))
      
      # result = np.concatenate(results, axis=0)
      #display_rgb("tonemapped", result)
      # save_rgb(f'tonemaps/test/{name[:-4]}', result)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()