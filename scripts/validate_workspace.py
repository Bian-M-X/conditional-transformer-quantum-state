#!/usr/bin/env python3
"""Lightweight repo and experiment metadata checks; requires stdlib only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=[
 'README.md','AGENTS.md','CONTRIBUTING.md','docs/STATE.md','docs/PROTOCOL.md',
 'docs/RESEARCH_LOG.md','docs/handoffs/_TEMPLATE.md','experiments/README.md',
 '.github/PULL_REQUEST_TEMPLATE.md','.github/ISSUE_TEMPLATE/research_task.md',
 '.github/workflows/collaboration-checks.yml']
REQUIRED_FIELDS={
 'schema_version','run_id','created_at_utc','status','owner','issue','git_commit',
 'seed','question','model','hamiltonian','parameter_split','command','environment',
 'budget','oracle_access','metrics','artifacts','notes'}
STATUS={'planned','running','completed','failed'}

def check_manifest(p):
    try: obj=json.loads(p.read_text(encoding='utf-8'))
    except (ValueError,OSError) as exc:return [f'{p}: invalid JSON: {exc}']
    errors=[]
    if not isinstance(obj,dict):return [f'{p}: manifest must be a JSON object']
    missing=REQUIRED_FIELDS-obj.keys()
    if missing:errors.append(f'{p}: missing fields: {sorted(missing)}')
    if obj.get('run_id')!=p.parent.name:errors.append(f'{p}: run_id must match run folder name')
    if obj.get('status') not in STATUS:errors.append(f'{p}: invalid status')
    if not isinstance(obj.get('seed'),int) or isinstance(obj.get('seed'),bool) or obj['seed']<0:errors.append(f'{p}: invalid seed')
    issue=obj.get('issue')
    local_id=obj.get('local_task_id','')
    local_ok=(issue is None and obj.get('schema_version',0)>=2 and obj.get('publication')=='local-only'
              and isinstance(local_id,str) and local_id.startswith('local-')
              and '/' not in local_id and '\\' not in local_id
              and (ROOT/'docs'/'local_tasks'/f'{local_id}.md').is_file())
    if not local_ok and (not isinstance(issue,int) or isinstance(issue,bool) or issue<=0):errors.append(f'{p}: invalid issue number or unregistered local task')
    if obj.get('status')=='completed':
        if not obj.get('command'):errors.append(f'{p}: completed run missing command')
        if not obj.get('environment'):errors.append(f'{p}: completed run missing environment')
        if not obj.get('metrics'):errors.append(f'{p}: completed run missing metrics')
    return errors

def validate():
    errors=[f'missing required file: {name}' for name in REQUIRED if not (ROOT/name).is_file()]
    manifests=sorted((ROOT/'experiments'/'runs').glob('*/manifest.json'))
    for p in manifests:errors.extend(check_manifest(p))
    return errors,len(manifests)

if __name__=='__main__':
    errors,count=validate()
    if errors:
        for e in errors:print('ERROR:',e,file=sys.stderr)
        sys.exit(1)
    print(f'PASS: required collaboration files present; {count} run manifest(s) validated')
