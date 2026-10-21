"""Reproduce the small quadratic forward/backward cost audit (not a timed benchmark)."""
import json
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
generator=torch.Generator().manual_seed(5)
matrix=torch.randn(64,64,generator=generator)
matrix=(matrix+matrix.T)/2
psi=torch.randn(64,generator=generator,requires_grad=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as profile:
    energy=psi@matrix@psi
    energy.backward()
rows=[dict(operation=e.key,count=e.count,shapes=e.input_shapes)
      for e in profile.key_averages(group_by_input_shape=True)
      if e.key in ('aten::mv','aten::mm','aten::matmul','aten::dot')]
assert sum(r['count'] for r in rows if r['operation']=='aten::mm' and r['shapes']==[[1,64],[64,64]])==2
print(json.dumps(rows,indent=2))
