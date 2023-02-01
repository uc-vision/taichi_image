from typing import Callable, Dict, List
import torch
import time

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
def benchmark(name,  func:Callable, args:List=None, kwargs:Dict=None, iterations:int=1, warmup:int=0):
  args = args or []
  kwargs = kwargs or {}
  
  for i in range(warmup):
    func(*args, **kwargs)

  with Benchmark(name, iterations) as b:
    for i in range(b.iterations):
      func(*args, **kwargs)
