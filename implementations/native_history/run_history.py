"""Add full final-history locality and recoverable weights to the frozen native runner."""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
spec=importlib.util.spec_from_file_location('native_runner',ROOT/'code/run_ke_scaling_v29.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
args=m.parse_args()
m.parse_args=lambda:args
original=m.evaluate_history
def history(samples,model,tokenizer,after_edits,max_samples):
    if after_edits!=args.n_edits:return original(samples,model,tokenizer,after_edits,max_samples)
    # Export every edited native matrix; the remaining parameters equal the base model.
    layers=m.parse_layer_list(args.target_layers);weights={}
    for name,p in model.named_parameters():
        match=re.search(r'(?:layers|h)\.(\d+)\.',name)
        if match and int(match.group(1)) in layers and re.search(r'(?:down_proj|c_proj)\.weight$',name):
            weights[name]=p.detach().cpu().clone()
    assert len(weights)==len(layers)
    path=args.output.parent/'edited_weights.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(weights,path)
    weight_record={'path':str(path),'sha256':m.sha256(path),'names':list(weights),'after_edits':after_edits}
    m.atomic_json(args.output.parent/'weights_manifest.json',weight_record)
    del weights
    h=original(samples,model,tokenizer,after_edits,len(samples))
    for row in h['evaluations']:
        sample=samples[row['history_position']]
        loc,n=m.relation_score(sample.get('locality',{}),model,tokenizer)
        row['retained_locality_target_consistency']=loc;row['retained_locality_n']=n
    vals=[r['retained_locality_target_consistency'] for r in h['evaluations'] if r['retained_locality_target_consistency'] is not None]
    assert len(vals)==len(samples), 'Missing locality targets must not silently reduce cohort'
    h.update(locality_target_consistency=100*m.aggregate(vals),n_locality_evaluated=len(vals),
        n_locality_probes=sum(r['retained_locality_n'] for r in h['evaluations']),
        locality_evaluator='existing relation_score/target_match, equal weight per historical edit',
        edited_weights=weight_record,
        archived_128_position_efficacy=100*m.aggregate([h['evaluations'][i]['retained_efficacy'] for i in m.sample_positions(len(samples),128)]))
    return h
m.evaluate_history=history
m.main()
r=json.loads(args.output.read_text());h=r['checkpoints'][-1]['history']
r['protocol']='fig4h_native250_history_v1'
r['final_history_locality_target_consistency']=h['locality_target_consistency']
r['edited_weights']=h['edited_weights']
m.atomic_json(args.output,r)
