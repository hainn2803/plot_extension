"""Fixed CUDA-library initialization, independently diagnosed without a model."""
import gc
import torch


def initialize(output, diagnosis):
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    from experiments.mcqa_c_at_it_gradual_pilot.engineering import device_memory
    evidence = read(diagnosis)
    if not evidence['completed'] or evidence['model_loaded'] or evidence['scientific_updates']:
        raise ValueError('Initialization requires the model-free diagnosis')
    expected = evidence['cold_to_initialized_shared_delta']
    if torch.cuda.is_initialized():
        raise ValueError('Library initialization must precede model loading')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    marker = torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    before = device_memory()
    a = torch.ones(32, 2304, device='cuda', requires_grad=True)
    b = torch.ones(2304, 128, device='cuda', requires_grad=True)
    y = a @ b
    assert torch.equal(y, torch.full_like(y, 2304))
    y.sum().backward()
    assert torch.equal(a.grad, torch.full_like(a, 128))
    ab, bb = a.detach().to(torch.bfloat16), b.detach().to(torch.bfloat16)
    yb = ab @ bb
    assert torch.equal(yb, torch.full_like(yb, 2304))
    for rank in (128, 256):
        x = torch.eye(2304, rank, device='cuda')
        q, r = torch.linalg.qr(x, mode='reduced')
        assert torch.allclose(q @ r, x, atol=2e-4, rtol=2e-4)
    del marker, a, b, y, ab, bb, yb, x, q, r
    gc.collect()
    torch.cuda.synchronize()
    after = device_memory()
    delta = after['shared_bytes_this_process'] - before['shared_bytes_this_process']
    result = dict(before=before, after=after, shared_delta=delta, diagnosed_delta=expected,
                  diagnosis=str(diagnosis), diagnosis_sha256=file_hash(diagnosis),
                  model_loaded=False, scientific_updates=0, rng_used=False,
                  original_memory_watch_unchanged=True, helper_sha256=file_hash(__file__))
    write(output / 'library_initialization.json', result)
    if delta != expected:
        raise RuntimeError('CUDA-library startup allocation differs from the independent diagnosis')
    if after['used_device_bytes_all_processes'] > after['physical_device_bytes']:
        raise RuntimeError('Device capacity check failed during CUDA-library initialization')
    return result
