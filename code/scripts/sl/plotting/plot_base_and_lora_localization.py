#!/usr/bin/env python3
"""Aggregate figures for the broad-base and LoRA-group localization screens."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spar.sl.config import load_config

ROOT=Path(__file__).resolve().parents[4]
RUNS=(("cat-47","code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"),("cat-48","code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml"),("cat-49","code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml"),("owl-47","code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml"),("owl-48","code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml"),("owl-49","code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml"))

def load(directory):
    out={}
    for name,path in RUNS:
        root=ROOT/load_config(ROOT/path)["experiment"]["output_root"]/directory/"summary.json"
        out[name]=json.loads(root.read_text())
    return out

def base():
    data=load("base_column_masking_v1"); out=ROOT/"artifacts/sl/base_column_masking_v1/aggregate"; out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),sharey=True,constrained_layout=True)
    for trait,ax in zip(("cat","owl"),axes):
        rows=[row for run,payload in data.items() if run.startswith(trait) for row in payload["rows"]]
        for method,color in (("magnitude_column","#1f77b4"),("taylor_contrastive_column","#2ca02c")):
            selected=defaultdict(list); random_rows=defaultdict(list)
            for row in rows:
                if row["method"]!=method: continue
                (random_rows if row["branch"]=="random" else selected)[row["density"]].append(row["sl_effect_pp"])
            x=sorted(selected); y=[np.mean(selected[v]) for v in x]; r=[np.mean(random_rows[v]) for v in x]; sd=[np.std(random_rows[v]) for v in x]
            ax.plot(np.array(x)*100,y,"o-",color=color,label=f"{method.replace('_column','')} selected")
            ax.plot(np.array(x)*100,r,"--",color=color,alpha=.65,label=f"{method.replace('_column','')} random")
            ax.fill_between(np.array(x)*100,np.array(r)-sd,np.array(r)+sd,color=color,alpha=.12)
        ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.25); ax.set(title=trait.title(),xlabel="Base input columns ablated per module (%)",ylabel="WS − WC preference (pp)"); ax.legend(fontsize=7)
    fig.suptitle("Broad base-column masking: selected versus module-matched random ablations",fontsize=13)
    fig.savefig(out/"base_column_masking_sl_effect.png",dpi=220); plt.close(fig)

def lora():
    data=load("lora_group_localization_v1"); out=ROOT/"artifacts/sl/lora_group_localization_v1/aggregate"; out.mkdir(parents=True,exist_ok=True)
    order=("full","early_only","middle_only","late_only","no_early","no_middle","no_late","attention_only","mlp_only","no_attention","no_mlp")
    fig,axes=plt.subplots(1,2,figsize=(14,4.6),sharey=True,constrained_layout=True)
    for trait,ax in zip(("cat","owl"),axes):
        by=defaultdict(list)
        for run,payload in data.items():
            if run.startswith(trait):
                by.update({})
                for row in payload["paired"]: by[row["condition"]].append(row["sl_effect_pp"])
        means=[np.mean(by[key]) for key in order]; sd=[np.std(by[key]) for key in order]
        ax.bar(np.arange(len(order)),means,yerr=sd,color=["#222222" if x=="full" else "#4c78a8" for x in order],capsize=3)
        ax.set(title=f"{trait.title()}: paired SL effect",xticks=np.arange(len(order)),xticklabels=[x.replace("_","\n") for x in order],ylabel="WS − WC preference (pp)"); ax.tick_params(axis="x",labelsize=7); ax.grid(axis="y",alpha=.25)
    fig.suptitle("LoRA layer and module localization (mean ± SD across seeds)",fontsize=13)
    fig.savefig(out/"lora_group_localization_sl_effect.png",dpi=220); plt.close(fig)

if __name__=="__main__": base(); lora()
