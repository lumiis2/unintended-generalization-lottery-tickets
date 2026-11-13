#!/usr/bin/env python3
"""Broad, reversible base-weight column masking baseline for subliminal learning.

Scores and masks are confined to each PEFT-wrapped base linear module.  A
selected column means zeroing ``W0[:, j]`` only at forward time; adapters and
checkpoint files are never changed.  Taylor scores are first-order scores of
the paired WS--WC loss sensitivity, collected without materialising gradients
for all 7B base parameters.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, random, re
from pathlib import Path

import torch

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion
from spar.sl.prompts import FAVORITE_ANIMAL_PROMPTS


def digest(rows): return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest()
def canon(name): return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")

def base_modules(model):
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module, "lora_A") and len(module.lora_A):
            base = module.base_layer
            if not hasattr(base, "weight") or base.weight.ndim != 2: raise ValueError(f"Expected Linear base layer: {name}")
            result[name] = base
    if not result: raise RuntimeError("No PEFT base linear modules found")
    return result

def layer_projection(name):
    match = re.search(r"\.layers\.(\d+)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", name)
    if not match: raise ValueError(name)
    return int(match.group(1)), match.group(2)

def encode(tokenizer, rows, max_length):
    encoded=[]
    for prompt, answer in rows:
        prefix=tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
        full=tokenizer.apply_chat_template([{"role":"user","content":prompt},{"role":"assistant","content":answer}], tokenize=False, add_generation_prompt=False)
        item=tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length); labels=list(item["input_ids"])
        n=min(len(tokenizer(prefix, add_special_tokens=False)["input_ids"]),len(labels)); labels[:n]=[-100]*n
        encoded.append({"input_ids":item["input_ids"],"labels":labels})
    return encoded

def batches(rows, pad, size):
    for start in range(0,len(rows),size):
        part=rows[start:start+size]; n=max(len(row["input_ids"]) for row in part)
        yield {"input_ids":torch.tensor([r["input_ids"]+[pad]*(n-len(r["input_ids"])) for r in part]), "attention_mask":torch.tensor([[1]*len(r["input_ids"])+[0]*(n-len(r["input_ids"])) for r in part]), "labels":torch.tensor([r["labels"]+[-100]*(n-len(r["labels"])) for r in part])}

class TaylorColumns:
    def __init__(self, model): self.modules=base_modules(model); self.sums={name:torch.zeros(base.weight.shape[1]) for name,base in self.modules.items()}; self.handles=[]
    def __enter__(self):
        for name, base in self.modules.items(): self.handles.append(base.register_forward_hook(self.hook(name)))
        return self
    def hook(self, name):
        def capture(module, inputs, output):
            x=inputs[0]; output.requires_grad_(True); weight=module.weight
            def accumulate(grad):
                # sum_{tokens,out} W[out,in] * dL/dy[out] * x[in]
                signed=(x.detach().float()*(grad.detach().float() @ weight.detach().float())).sum(dim=(0,1)).cpu()
                self.sums[name].add_(signed)
            output.register_hook(accumulate)
        return capture
    def __exit__(self,*_):
        for handle in self.handles: handle.remove()

def signed_taylor(cfg, adapter, rows, batch_size):
    tokenizer=load_tokenizer(cfg); model=load_model(cfg,adapter=str(adapter),trainable=True)
    # Make residual activations differentiable while keeping base weights frozen.
    model.enable_input_require_grads()
    for name,param in model.named_parameters(): param.requires_grad_(".lora_" in name)
    encoded=encode(tokenizer, rows, cfg["training"]["max_length"])
    try:
        with TaylorColumns(model) as capture:
            for batch in batches(encoded, tokenizer.pad_token_id, batch_size):
                model.zero_grad(set_to_none=True); out=model(**{k:v.to(model.device) for k,v in batch.items()}); out.loss.backward()
        return capture.sums
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def heldout_numeric(cfg, count, seed):
    root=Path(cfg["experiment"]["output_root"])/"datasets/control"; used={r["source_attempt_id"] for r in read_jsonl(root/"train.jsonl")}
    rows=[(r["prompt"],r["completion"]) for r in read_jsonl(root/"raw.jsonl") if r["parse"]["valid"] and r["attempt_id"] not in used]
    random.Random(seed).shuffle(rows); return rows[:count]

def scores(cfg, root, out, trait_n, numeric_n, seed):
    ws=root/"models/trait/adapter"; wc=root/"models/control/adapter"
    trait=[(p,cfg["trait"]["target"]) for p in FAVORITE_ANIMAL_PROMPTS[:trait_n]]; numeric=heldout_numeric(cfg,numeric_n,seed)
    ws_t=signed_taylor(cfg,ws,trait,1); wc_t=signed_taylor(cfg,wc,trait,1)
    ws_n=signed_taylor(cfg,ws,numeric,1); wc_n=signed_taylor(cfg,wc,numeric,1)
    trait_pair={k:(ws_t[k]-wc_t[k]).abs() for k in ws_t}; numeric_pair={k:(ws_n[k]-wc_n[k]).abs() for k in ws_n}
    tm=torch.cat(list(trait_pair.values())).mean().clamp_min(1e-12); nm=torch.cat(list(numeric_pair.values())).mean().clamp_min(1e-12)
    contrast={k:(trait_pair[k]/tm-numeric_pair[k]/nm).clamp_min(0) for k in trait_pair}
    control_trait={k:wc_t[k].abs() for k in wc_t}; control_numeric={k:wc_n[k].abs() for k in wc_n}
    ctm=torch.cat(list(control_trait.values())).mean().clamp_min(1e-12); cnm=torch.cat(list(control_numeric.values())).mean().clamp_min(1e-12)
    control_contrast={k:(control_trait[k]/ctm-control_numeric[k]/cnm).clamp_min(0) for k in control_trait}
    reference=load_model(cfg,adapter=str(ws)).eval()
    try:
        magnitude={name:base.weight.detach().float().norm(dim=0).cpu() for name,base in base_modules(reference).items()}
    finally:
        del reference; gc.collect(); torch.cuda.empty_cache()
    payload={"magnitude_column":magnitude,"taylor_contrastive_column":contrast,"taylor_control_contrastive_column":control_contrast}
    out.mkdir(parents=True,exist_ok=True)
    for method,value in payload.items(): torch.save({"scores":value,"metadata":{"method":method,"granularity":"base_input_column","trait_examples":trait_n,"numeric_examples":numeric_n,"seed":seed}},out/f"{method}.pt")
    return payload

def selected_units(score, density, random_seed=None):
    result={}; g=torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    for name, values in score.items():
        n=int(round(values.numel()*density)); mask=torch.zeros(values.numel(),dtype=torch.bool)
        if n: mask[(torch.randperm(values.numel(),generator=g)[:n] if g else torch.argsort(values,descending=True,stable=True)[:n])]=True
        result[name]=mask
    return result

def apply_base_mask(model, units):
    """Apply a real, reversible base-column intervention.

    Boolean advanced indexing returns a temporary tensor in PyTorch, so this
    must use ``index_fill_`` rather than ``weight[:, mask].zero_()``.
    """
    modules=base_modules(model); missing=set(units)-set(modules)
    if missing: raise KeyError(sorted(missing)[:3])
    # ``units`` can intentionally name only a targeted subset of modules.
    # Modules omitted from it remain exactly intact.
    modules={name: modules[name] for name in units}
    state=[]
    with torch.no_grad():
        for name,base in modules.items():
            indices=torch.nonzero(units[name],as_tuple=False).flatten().to(base.weight.device)
            original=base.weight.index_select(1,indices).detach().clone()
            base.weight.index_fill_(1,indices,0)
            state.append((name,base,indices,original))
    return state

def restore_base_mask(state):
    with torch.no_grad():
        for _,base,indices,original in state:
            base.weight.index_copy_(1,indices,original.to(base.weight.device,base.weight.dtype))

def lora_snapshot(model):
    return {name:param.detach().clone() for name,param in model.named_parameters() if ".lora_" in name}

def validate_base_mask(model, units):
    """Verify selection, preservation, model-output change, and restoration."""
    modules={name: base for name, base in base_modules(model).items() if name in units}
    if not modules:
        raise ValueError("No targeted base modules supplied for validation")
    lora_before=lora_snapshot(model); checks={}
    for name,base in modules.items():
        selected=torch.nonzero(units[name],as_tuple=False).flatten().to(base.weight.device)
        unselected=torch.nonzero(~units[name],as_tuple=False).flatten().to(base.weight.device)
        if selected.numel()==0 or unselected.numel()==0: raise ValueError(f"Invalid support in {name}")
        probe=unselected[:min(8,unselected.numel())]
        checks[name]=(base.weight.index_select(1,selected).detach().clone(),base.weight.index_select(1,probe).detach().clone(),probe)
    probe_name,probe=next(iter(modules.items()))
    x=torch.randn(2,3,probe.weight.shape[1],device=probe.weight.device,dtype=probe.weight.dtype)
    with torch.no_grad(): baseline=probe(x).detach().float().cpu()
    state=apply_base_mask(model,units)
    try:
        with torch.no_grad(): masked=probe(x).detach().float().cpu()
        if torch.equal(baseline,masked): raise AssertionError(f"Masked forward output did not change for {probe_name}")
        for name,base in modules.items():
            selected_before,unselected_before,unselected=checks[name]
            selected=torch.nonzero(units[name],as_tuple=False).flatten().to(base.weight.device)
            if torch.count_nonzero(base.weight.index_select(1,selected)).item()!=0: raise AssertionError(f"Selected columns not zero in {name}")
            if not torch.equal(base.weight.index_select(1,unselected),unselected_before): raise AssertionError(f"Unselected columns changed in {name}")
        for name,param in model.named_parameters():
            if name in lora_before and not torch.equal(param.detach(),lora_before[name]): raise AssertionError(f"LoRA changed: {name}")
    finally:
        restore_base_mask(state)
    with torch.no_grad(): restored=probe(x).detach().float().cpu()
    if not torch.equal(baseline,restored): raise AssertionError(f"Forward output not restored for {probe_name}")
    for name,base in modules.items():
        selected_before,_,_=checks[name]; selected=torch.nonzero(units[name],as_tuple=False).flatten().to(base.weight.device)
        if not torch.equal(base.weight.index_select(1,selected),selected_before): raise AssertionError(f"Weights not restored in {name}")
    return {"modules_checked":len(modules),"probe_module":probe_name,"forward_changed":True,"restored":True,"lora_unchanged":True}

def preflight_units(model, density=.01):
    result={}
    for name,base in base_modules(model).items():
        count=max(1,round(base.weight.shape[1]*density)); mask=torch.zeros(base.weight.shape[1],dtype=torch.bool); mask[:count]=True; result[name]=mask
    return result

def run_preflight(cfg, root):
    result={"adapters":{}}
    for label,path in (("ws",root/"models/trait/adapter"),("wc",root/"models/control/adapter")):
        model=load_model(cfg,adapter=str(path)).eval()
        try: result["adapters"][label]=validate_base_mask(model,preflight_units(model))
        finally:
            del model; gc.collect(); torch.cuda.empty_cache()
    return result

def preference(model,tokenizer,cfg,rows,samples,seed):
    torch.manual_seed(seed); expanded=[r for r in rows for _ in range(samples)]; outputs=batched_generate(model,tokenizer,[render_chat(tokenizer,r["prompt"]) for r in expanded],cfg["evaluation"]); pat=re.compile(rf"\b{re.escape(cfg['trait']['target'])}s?\b",re.I)
    raw=[{"prompt_id":r.get("id",i//samples),"sample":i%samples,"output":o,"target_hit":bool(pat.search(o))} for i,(r,o) in enumerate(zip(expanded,outputs,strict=True))]
    return {"n":len(raw),"target_rate":sum(r["target_hit"] for r in raw)/len(raw)},raw

def numeric_eval(model,tokenizer,cfg,condition,count,seed,offset=0):
    root=Path(cfg["experiment"]["output_root"])/"datasets"/condition
    used={r["source_attempt_id"] for r in read_jsonl(root/"train.jsonl")}
    rows=[r for r in read_jsonl(root/"raw.jsonl") if r["parse"]["valid"] and r["attempt_id"] not in used]
    random.Random(seed).shuffle(rows)
    if len(rows) < offset + count:
        raise ValueError(f"Only {len(rows)} held-out numeric examples; need {offset + count}")
    rows=rows[offset:offset+count]
    torch.manual_seed(seed+1)
    outputs=batched_generate(model,tokenizer,[render_chat(tokenizer,r["prompt"]) for r in rows],cfg["generation"])
    valid=[parse_number_completion(o,minimum=cfg["data"]["integer_min"],maximum=cfg["data"]["integer_max"],max_values=cfg["data"]["max_added_values"]).valid for o in outputs]
    return {"n":len(outputs),"valid_rate":sum(valid)/len(valid),"offset":offset,"seed":seed},[{"source_attempt_id":r["attempt_id"],"output":o,"valid":v} for r,o,v in zip(rows,outputs,valid,strict=True)]

@torch.inference_mode()
def generic_loss(model,tokenizer,texts):
    vals=[]
    for text in texts:
        ids=tokenizer(text,return_tensors="pt",truncation=True,max_length=256).input_ids.to(model.device)
        if ids.shape[1]>1: vals.append(float(model(input_ids=ids,labels=ids).loss))
    return {"n":len(vals),"mean_nll":sum(vals)/len(vals),"perplexity":float(torch.exp(torch.tensor(vals).mean()))}

def evaluate_adapter(cfg,adapter,condition,units,prompts,generic,args):
    tokenizer=load_tokenizer(cfg); model=load_model(cfg,adapter=str(adapter)).eval()
    state=None
    try:
        state=apply_base_mask(model,units); pref,raw=preference(model,tokenizer,cfg,prompts,args.samples_per_prompt,args.seed); num,nraw=numeric_eval(model,tokenizer,cfg,condition,args.numeric_examples,args.seed); return {"preference":pref,"numeric":num,"generic_loss":generic_loss(model,tokenizer,generic)},raw,nraw
    finally:
        if state is not None: restore_base_mask(state)
        del model; gc.collect(); torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--final-root",type=Path,default=Path("experiments/subliminal_learning/frozen_evaluation/v1")); p.add_argument("--output-name",default="base_column_masking_v2"); p.add_argument("--preflight-only",action="store_true"); p.add_argument("--densities",type=float,nargs="+",default=[.01,.03,.10]); p.add_argument("--random-replicates",type=int,default=3); p.add_argument("--samples-per-prompt",type=int,default=10); p.add_argument("--numeric-examples",type=int,default=128); p.add_argument("--generic-examples",type=int,default=64); p.add_argument("--trait-examples",type=int,default=32); p.add_argument("--score-numeric-examples",type=int,default=64); p.add_argument("--seed",type=int,default=20261110); args=p.parse_args()
    cfg=load_config(args.config); root=Path(cfg["experiment"]["output_root"]); out=root/args.output_name; out.mkdir(parents=True,exist_ok=True)
    if args.preflight_only:
        write_json(out/"preflight.json",run_preflight(cfg,root)); return
    score_dir=out/"scores"; manifest=json.loads((args.final_root/"manifest.json").read_text()); prompts=read_jsonl(args.final_root/manifest["trait_prompts"]["path"])
    if digest(prompts)!=manifest["trait_prompts"]["sha256"]: raise ValueError("Frozen trait prompts failed hash check")
    generic=[r["text"] for r in read_jsonl(args.final_root/manifest["generic_loss"]["frozen"]["path"])[:args.generic_examples]]
    score_map=scores(cfg,root,score_dir,args.trait_examples,args.score_numeric_examples,args.seed)
    for method in ("magnitude_column","taylor_contrastive_column"):
        probe=selected_units(score_map[method],.03)
        for path in (root/"models/trait/adapter",root/"models/control/adapter"):
            model=load_model(cfg,adapter=str(path)).eval()
            try: validate_base_mask(model,probe)
            finally:
                del model; gc.collect(); torch.cuda.empty_cache()
    specs=[]
    for method in ("magnitude_column","taylor_contrastive_column"):
        for density in args.densities:
            specs.append((method,density,"selected",None))
            specs += [(method,density,"random",args.seed+1000*rep+round(density*100)) for rep in range(args.random_replicates)]
    # Diagnostic: selector obtained from WC alone, at the middle density.
    specs.append(("taylor_control_contrastive_column",.03,"selected",None))
    all_rows=[]
    for method,density,branch,rseed in specs:
        identifier=f"{method}_d{int(density*100):02d}_{branch}"+(f"_r{rseed}" if rseed is not None else "")
        path=out/"results"/f"{identifier}.json"
        if path.exists(): all_rows.append(json.loads(path.read_text())); continue
        units=selected_units(score_map[method],density,rseed); ws,wsraw,wsnum=evaluate_adapter(cfg,root/"models/trait/adapter","trait",units,prompts,generic,args); wc,wcraw,wcnum=evaluate_adapter(cfg,root/"models/control/adapter","control",units,prompts,generic,args)
        row={"identifier":identifier,"method":method,"density":density,"branch":branch,"random_seed":rseed,"selected_columns":sum(int(v.sum()) for v in units.values()),"total_columns":sum(v.numel() for v in units.values()),"ws":ws,"wc":wc,"sl_effect_pp":100*(ws["preference"]["target_rate"]-wc["preference"]["target_rate"])}
        write_json(path,row); write_jsonl(out/"raw"/f"{identifier}_ws_trait.jsonl",wsraw); write_jsonl(out/"raw"/f"{identifier}_wc_trait.jsonl",wcraw); write_jsonl(out/"raw"/f"{identifier}_ws_numeric.jsonl",wsnum); write_jsonl(out/"raw"/f"{identifier}_wc_numeric.jsonl",wcnum); all_rows.append(row)
    write_json(out/"summary.json",{"config":str(args.config),"final_root":str(args.final_root),"densities":args.densities,"random_replicates":args.random_replicates,"rows":all_rows})

if __name__=="__main__": main()
