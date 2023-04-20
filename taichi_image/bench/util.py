from typing import Callable, Dict, List
import torch
import time
from tqdm import tqdm

from typeguard import typechecked

class Benchmark:
  @typechecked
  def __init__(self, name:str, iterations:int=1):
    self.iterations = iterations
    self.name = name

  def __enter__(self):
    torch.cuda.synchronize()
    self.start = time.time()
    return self

  def __exit__(self, type, value, traceback):
    torch.cuda.synchronize()
    self.elapsed = time.time() - self.start

    if self.iterations > 1:
      print(
          f"{self.name}: {self.elapsed:.4f}s {self.iterations / self.elapsed:.2f} it/s"
      )
    else:
      print(f"{self.name}: {self.elapsed:.4f}s")

@typechecked
def benchmark(name,  func:Callable, args:List=None, kwargs:Dict=None, iterations:int=1, warmup:int=0, progress=tqdm):
  args = args or []
  kwargs = kwargs or {}
  if progress is None:
    progress = lambda x: x
  
  print(f"Warming up {name} for {warmup} iterations...")

  for i in progress(range(warmup)):
    func(*args, **kwargs)
  
  print(f"Running {name} for {iterations} iterations...")



  with Benchmark(name, iterations) as b:
    for i in progress(range(b.iterations)):
      func(*args, **kwargs)
