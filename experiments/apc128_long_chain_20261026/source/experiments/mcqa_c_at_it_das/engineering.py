"""Measured full-backbone gradients and accumulation, without changing arithmetic."""
import gc
import time

import numpy as np
import torch

from experiments.mcqa_c_at_localization import backend
from experiments.mcqa_c_at_localization.lens import load_lens
from experiments.mcqa_c_at_localization.reference import jonathan_patch
from experiments.mcqa_c_at_localization.protocol import LABELS
from experiments.mcqa_c_at_localization.storage import read, write, file_hash, digest
from experiments.mcqa_c_at_prompt_diagnostic.runner import numeric_check
from .protocol import ids_for, parent
from .training import optimizer_step


def source_vectors(model, source, layer, position, batch):
    values=[]
    with torch.no_grad():
        for start in range(0,len(source),batch):
            logits, hidden = backend.forward(model, source[start:start+batch].to("cuda"), capture=[layer])
            values.append(hidden[layer][:,position].detach().cpu().clone())
            del logits,hidden
    return torch.cat(values)


def check(name, left, right):
    a, b = left.detach(), right.detach()
    passed = torch.allclose(a,b,atol=2e-4,rtol=2e-4)
    result = {"name":name, "passed":passed, "max_abs_difference":float((a.float()-b.float()).abs().max())}
    if not passed:
        raise RuntimeError("Numerical mismatch: " + name + "; atol=rtol=2e-4 unchanged")
    return result


def real_gradients(model, ids, sources, targets, layer, position, rank, batch, session):
    torch.manual_seed(1)
    subspace = backend.LearnedSubspace(2304,rank).to("cuda")
    inputs, source = ids[:batch].to("cuda"), sources[:batch].to("cuda")
    logits, _ = backend.forward(model,inputs,[backend.Patch(layer,position,source,subspace.basis())])
    loss = torch.nn.functional.cross_entropy(logits,targets[:batch].to("cuda"))
    direct = torch.autograd.grad(loss,subspace.raw)[0].detach()
    del logits,loss
    with torch.no_grad():
        _, hidden = backend.forward(model,inputs,capture=[layer])
        base = hidden[layer][:,position].detach()
    q = subspace.basis()
    trusted = backend.same_shape_reference(base,source,q,jonathan_patch)
    logits, _ = backend.forward(model,inputs,[backend.Patch(layer,position,trusted)])
    loss = torch.nn.functional.cross_entropy(logits,targets[:batch].to("cuda"))
    reference = torch.autograd.grad(loss,subspace.raw)[0].detach()
    checks = [check("real_model_centered_Jonathan_basis_gradient",direct,reference)]
    del logits,loss,trusted,q,hidden,base,direct,reference
    individual = []
    for start in (0,batch):
        session.check()
        logits,_ = backend.forward(model,ids[start:start+batch].to("cuda"),
            [backend.Patch(layer,position,sources[start:start+batch],subspace.basis())])
        loss = torch.nn.functional.cross_entropy(logits,targets[start:start+batch].to("cuda"),reduction="sum")/(2*batch)
        individual.append(torch.autograd.grad(loss,subspace.raw)[0].detach())
        del logits,loss
    expected = individual[0]+individual[1]
    subspace.zero_grad(set_to_none=True)
    for start in (0,batch):
        logits,_ = backend.forward(model,ids[start:start+batch].to("cuda"),
            [backend.Patch(layer,position,sources[start:start+batch],subspace.basis())])
        loss = torch.nn.functional.cross_entropy(logits,targets[start:start+batch].to("cuda"),reduction="sum")/(2*batch)
        loss.backward()
        del logits,loss
    checks.append(check("real_model_matched_shape_accumulation",subspace.raw.grad,expected))
    assert subspace.raw.grad.norm()>0, "Gradient test must be nontrivial"
    return checks


def time_site(model, ids, source_ids, targets, layer, position, rank, batch, session, repeats=3):
    sources = source_vectors(model,source_ids,layer,position,batch)
    torch.manual_seed(1)
    subspace = backend.LearnedSubspace(2304,rank).to("cuda")
    optimizer = torch.optim.Adam(subspace.parameters(),lr=.01)
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer_step(model,subspace,optimizer,ids,sources,targets,torch.arange(32),layer,position,batch,session)
        torch.cuda.synchronize()
        values.append(time.monotonic()-start)
    with torch.no_grad():
        timings = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.monotonic()
            logits,_ = backend.forward(model,ids[:batch].to("cuda"),[backend.Patch(layer,position,sources[:batch],subspace.basis())])
            assert torch.isfinite(logits).all()
            del logits
            torch.cuda.synchronize()
            timings.append(time.monotonic()-start)
    return {"layer":layer,"position":position,"rank":rank,"batch":batch,
            "effective_update_seconds":float(np.median(values)),"calibration_batch_seconds":float(np.median(timings)),
            "update_samples":values,"peak_reserved_bytes":torch.cuda.max_memory_reserved()}


def profile(model,root,session,progress):
    reg = read(root/"registration.json")
    rows = read(root/"engineering/profile_rows.json")
    ids, source = ids_for(root,"train",rows), ids_for(root,"train",rows,"source")
    labels = read(root/"token_map.json")["label_ids"]
    targets = torch.tensor([labels[LABELS.index(r["targets"]["C"])] for r in rows])
    pool = read(root/"candidate_pool.json")
    earliest = min(pool,key=lambda c:(c["layer"],c["position"]))
    layers = sorted({(c["layer"],c["position"]) for c in pool})
    samples=[]
    total = torch.cuda.get_device_properties(0).total_memory
    for batch in (1,2,4,8,16,32):
        session.check()
        if batch>=16:
            samples.append({"batch":batch,"status":"not_attempted_inference_already_exceeds_dedicated_memory",
                            "evidence":file_hash(parent.ROOT/"engineering/profile.json")})
            continue
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            measured = time_site(model,ids,source,targets,earliest["layer"],earliest["position"],512,batch,session)
            measured["device_bytes"] = total
            measured["status"] = "passed" if measured["peak_reserved_bytes"]<=total else "exceeds_dedicated_memory"
        except torch.cuda.OutOfMemoryError:
            measured={"batch":batch,"status":"CUDA_OOM_not_validated"}
        samples.append(measured)
        gc.collect()
        torch.cuda.empty_cache()
        write(root/"engineering/training_profile_progress.json",{"batches":samples})
        progress(root,"training_profile",batch=batch,result=measured)
    passing=[s for s in samples if s["status"]=="passed"]
    assert passing,"No physical training batch passed"
    batch=max(s["batch"] for s in passing)
    with torch.no_grad():
        native = numeric_check(model,ids[:batch].to("cuda"),"IT_DAS_training_shape",session,torch,backend)
    matrices,_=load_lens(parent.LENS_PATH)
    numerical=parent.numerical(model,root,ids,source,batch,read(root/"token_map.json"),matrices,session)
    del matrices
    sources=source_vectors(model,source,earliest["layer"],earliest["position"],batch)
    gradients=real_gradients(model,ids,sources,targets,earliest["layer"],earliest["position"],512,batch,session)
    write(root/"engineering/backward_numerical.json",{"passed":True,"checks":gradients,"native":native,"atol":2e-4,"rtol":2e-4},immutable=True)
    times=[]
    for layer,position in layers:
        torch.cuda.reset_peak_memory_stats()
        timed=time_site(model,ids,source,targets,layer,position,512,batch,session)
        assert timed["peak_reserved_bytes"]<=total, "Selected training batch spills into shared memory"
        times.append(timed)
        gc.collect()
        torch.cuda.empty_cache()
    # Repeated early-layer backward work exercises the same thermal policy as the sweep.
    sustained_subspace=backend.LearnedSubspace(2304,512).to("cuda")
    sustained_optimizer=torch.optim.Adam(sustained_subspace.parameters(),lr=.01)
    start=time.monotonic()
    sustained_updates=[]
    while time.monotonic()-start<180:
        session.check()
        torch.cuda.synchronize()
        tick=time.monotonic()
        optimizer_step(model,sustained_subspace,sustained_optimizer,ids,sources,targets,torch.arange(32),
                       earliest["layer"],earliest["position"],batch,session)
        torch.cuda.synchronize()
        sustained_updates.append(time.monotonic()-tick)
    times[0]["effective_update_seconds"]=max(times[0]["effective_update_seconds"],float(np.quantile(sustained_updates,.9)))
    by_site={(t["layer"],t["position"]):t for t in times}
    per_configuration={}
    for site,t in by_site.items():
        per_configuration[str(site)]=20*(48*t["effective_update_seconds"]+(768//batch)*t["calibration_batch_seconds"])
    search=sum(per_configuration[str((c["layer"],c["position"]))] for c in pool)
    repeat=4*max(per_configuration.values())
    cache=4*(1536+768+512+256)/batch*max(t["calibration_batch_seconds"] for t in times)
    seconds=search+repeat+cache
    result={"registration":digest(reg),"batch":batch,"batches":samples,"site_times":times,
            "components_seconds":{"seed1_28_configurations":search,"four_repeat_upper_bound":repeat,"cache_readouts_and_lock_overhead":cache},
            "hours_with_25_percent_allowance":seconds*1.25/3600,"calibration_included":True,"epoch_checkpoint_overhead":"covered by allowance",
            "numerical":file_hash(root/"engineering/numerical.json"),"backward_numerical":file_hash(root/"engineering/backward_numerical.json"),
            "sustained_seconds":time.monotonic()-start,"safe_for_local_launch":seconds*1.25<=12*3600,
            "scope":"28 seed1 trainings, up to four repeats, independent readout fitting and calibration; heldout evaluation not included"}
    write(root/"engineering/training_profile.json",result,immutable=True)
    progress(root,"profile_complete",**{k:result[k] for k in ("batch","hours_with_25_percent_allowance","safe_for_local_launch")})
    return result
