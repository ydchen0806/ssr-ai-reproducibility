"""Resolve a recorded invocation without changing its scientific arguments."""
from pathlib import Path
import argparse,json,os,subprocess,sys,shlex
p=argparse.ArgumentParser();p.add_argument('invocation',type=Path);p.add_argument('--path-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
record=json.loads(args.invocation.read_text());mapping=json.loads(args.path_map.read_text())
assert isinstance(mapping,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in mapping.items())
def resolve(value):
 value=str(value)
 for old,new in sorted(mapping.items(),key=lambda x:-len(x[0])):
  if value==old or value.startswith(old.rstrip('/')+'/'):return str(Path(new).expanduser())+value[len(old):]
 return value
command=[resolve(v) for v in record.get('command',record.get('argv',[]))];assert len(command)>1,'Missing recorded command';command[0]=sys.executable
for flag in ['--output','--output-dir']:
 if flag in command:
  i=command.index(flag)+1;command[i]=str(args.output.resolve()/('results.json' if Path(command[i]).suffix=='.json' else ''))
env={k:resolve(v) for k,v in record.get('env',{}).items()};driver=Path(command[1]);root=driver.parent.parent if driver.parent.name=='scripts' else driver.parent
if (root/'llm_ke').exists():env['SSR_REPO_DIR']=str(root)
unresolved=[v for v in command[1:]+list(env.values()) if v.startswith('/unify/')]
if unresolved:p.error('Unresolved original-machine paths; extend --path-map for every model, code, stream and cache location.')
print(json.dumps({'command':command,'environment_overrides':env},indent=2))
if args.execute:
 if args.output.exists() and any(args.output.iterdir()):p.error('Output directory must be empty for a fresh replay.')
 args.output.mkdir(parents=True,exist_ok=True)
 subprocess.run(command,env={**os.environ,**env},check=True)
