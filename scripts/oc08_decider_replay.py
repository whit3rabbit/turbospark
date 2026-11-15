#!/usr/bin/env python3
"""Replay saved greedy model replies through the CURRENT pre-filter.

A result file made with oc08_decider_eval.py --all holds the model reply for
every item, and greedy replies do not depend on the pre-filter. So a gate
change can be re-scored offline without a model:
  oc08_decider_replay.py run1.json [run2.json] [--no-second-person]
Only valid while the rubric and the user-turn shape are unchanged. The gate
lists are read from the Swift source by oc08_decider_eval.py.
"""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
flag = '--no-second-person' in sys.argv
import oc08_decider_eval as m
src=m.swift_source()
files=[a for a in sys.argv[1:] if not a.startswith('--')]
runs=[]
for f in files:
    rows=[r for r in json.load(open(f))['rows'] if not r.get('ambiguous')]
    out=[]
    for r in rows:
        pre=m.prefilter(r['text'],src)
        mr=r['model_reply']
        out.append({**r,'pre':pre,'sent':(not pre) and mr=='REQUEST'})
    runs.append(out)
for i,out in enumerate(runs):
    sent=[r for r in out if r['sent']]; good=[r for r in sent if r['should_send']]
    pos=[r for r in out if r['should_send']]
    lo,hi=m.wilson(len(good),len(sent))
    rl,rh=m.wilson(len(good),len(pos))
    print(f"run{i+1}: n={len(out)} sent={len(sent)} correct={len(good)} prec={len(good)/max(1,len(sent)):.3f} W95=[{lo:.3f},{hi:.3f}] recall={len(good)}/{len(pos)}={len(good)/len(pos):.3f} [{rl:.3f},{rh:.3f}]")
    from collections import Counter
    c=Counter(r['group'] for r in sent if not r['should_send']); print("  false allows by class:",dict(c))
    if i==0 or True:
        for r in sent:
            if not r['should_send']: print("   FA",r['group'],'|',r['text'])
    held=[r for r in pos if r['pre']]; print("  safe requests held by gate:",len(held)); 
    miss=[r for r in pos if not r['pre'] and not r['sent']]; print("  safe requests passed gate but model not REQUEST:",len(miss), [ (r['model_reply'],r['text'][:40]) for r in miss])
    print("  reaches model per class:",dict(Counter(r['group'] for r in out if not r['pre'])))
if len(runs)==2:
    a,b=runs
    print("agreement model reply:",sum(x['model_reply']==y['model_reply'] for x,y in zip(a,b)),"/",len(a),"sent:",sum(x['sent']==y['sent'] for x,y in zip(a,b)))
