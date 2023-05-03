import taichi as ti
import argparse
import torch


def add_taichi_args(parser):
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--n", type=int, default=100000)
  parser.add_argument("--log", default=ti.INFO, choices=ti._logging.supported_log_levels)
  parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
  parser.add_argument("--show", action="store_true")


def init_with_args(**kwargs):
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)


  parser = argparse.ArgumentParser()
  parser.add_argument("image", type=str)
  add_taichi_args(parser)

  args = parser.parse_args()

  ti.init(debug=args.debug, 
    arch=ti.cuda if args.device == "cuda" else ti.cpu,  log_level=args.log, **kwargs)

  return args