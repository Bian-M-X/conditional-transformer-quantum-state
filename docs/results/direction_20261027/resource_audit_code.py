"""Preserve the two stdin benchmark bodies actually executed in this audit.

The INITIAL and WARMED strings below reproduce the executed Python bodies.
This dispatcher was added afterwards and has not itself been executed. The
original PowerShell commands set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1
before starting .venv/Scripts/python.exe. Run from the repository root.
No training runs, files, dependency changes or scientific conclusions result.

Usage: .venv/Scripts/python.exe tmp/issue17_resource_audit.py initial|warmed
"""
import os
import sys

INITIAL = r'''
import os,sys,time,json,ctypes,platform
import numpy as np,scipy.linalg,torch
sys.path.insert(0,'src')
from ctqs.physics import basis,hamiltonian
from ctqs.models import ConditionalTransformer
from ctqs.exact_sr import amplitude_jacobian
class MemoryStatus(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total_phys',ctypes.c_ulonglong),('avail_phys',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),('total_virtual',ctypes.c_ulonglong),('avail_virtual',ctypes.c_ulonglong),('avail_extended',ctypes.c_ulonglong)]
m=MemoryStatus();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print(json.dumps({'RAM_GiB':m.total_phys/2**30,'available_RAM_GiB':m.avail_phys/2**30,'CPU':platform.processor(),'logical_cpu':os.cpu_count(),'torch':torch.__version__,'cuda':torch.cuda.is_available()}),flush=True)
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
for n in [6,8,10]:
 d=2**n;t=time.perf_counter();H=hamiltonian(n,.4,.02);build=time.perf_counter()-t
 rng=np.random.default_rng(1);x=rng.normal(size=d);ix=np.arange(d);z=1-2*basis(n);diag=-np.sum(z[:,:-1]*z[:,1:],axis=1)+.02*z[:,0]
 def sparse_action(v):
  out=diag*v
  for i in range(n):out-=.4*v[ix^(1<<i)]
  return out
 assert np.allclose(H@x,sparse_action(x),atol=1e-12)
 t=time.perf_counter()
 for _ in range(10):y=H@x
 dense_ms=100*(time.perf_counter()-t)
 t=time.perf_counter()
 for _ in range(10):y=sparse_action(x)
 sparse_ms=100*(time.perf_counter()-t)
 t=time.perf_counter();ev=scipy.linalg.eigh(H,subset_by_index=[0,1],eigvals_only=True);ed=time.perf_counter()-t
 torch.manual_seed(2);model=ConditionalTransformer(n).double();bits=torch.tensor(basis(n),dtype=torch.float64);Ht=torch.tensor(H)
 t=time.perf_counter()
 for _ in range(3):
  model.zero_grad();p=model.psi(bits,[.4,.02]);(p@Ht@p/n).backward()
 closure=(time.perf_counter()-t)/3
 t=time.perf_counter();p,J=amplitude_jacobian(model,bits,[.4,.02]);jac=time.perf_counter()-t
 print(json.dumps({'n':n,'d':d,'parameters':sum(p.numel() for p in model.parameters()),'H_MiB':H.nbytes/2**20,'H_build_s':build,'dense_H_action_ms':dense_ms,'matrix_free_H_action_ms':sparse_ms,'ED_first2_s':ed,'one_condition_closure_s':closure,'one_condition_J_s':jac,'J_MiB':J.numel()*J.element_size()/2**20}),flush=True)
'''

WARMED = r'''
import json,time,sys,torch
sys.path.insert(0,'src')
from ctqs.models import ConditionalTransformer
from ctqs.physics import basis,hamiltonian
from ctqs.exact_sr import amplitude_jacobian
import numpy as np
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
warm=ConditionalTransformer(3).double();amplitude_jacobian(warm,torch.tensor(basis(3),dtype=torch.float64),[.4,.02])
for n in [6,8,10]:
 model=ConditionalTransformer(n).double();bits=torch.tensor(basis(n),dtype=torch.float64);H=torch.tensor(hamiltonian(n,.4,.02));ts=[];tj=[]
 for _ in range(3):
  t=time.perf_counter();model.zero_grad();p=model.psi(bits,[.4,.02]);(p@H@p/n).backward();ts.append(time.perf_counter()-t)
  t=time.perf_counter();p,J=amplitude_jacobian(model,bits,[.4,.02]);tj.append(time.perf_counter()-t)
 A=J.repeat(3,1)/(3**.5);b=torch.randn(len(A));t=time.perf_counter();K=A@A.T;K.diagonal().add_(1e-4);dual=torch.linalg.solve(K,b);kernel=time.perf_counter()-t
 print(json.dumps({'n':n,'closure_median_s':np.median(ts),'J_median_s':np.median(tj),'three_condition_kernel_build_solve_s':kernel,'kernel_MiB':K.numel()*8/2**20,'warning':'kernel is synthetic repeated J; timing only, not training or fidelity result'}),flush=True)
'''

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('initial', 'warmed'):
        raise SystemExit('Select initial or warmed; run from repository root.')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    exec(INITIAL if sys.argv[1] == 'initial' else WARMED, {'__name__': '__main__'})
