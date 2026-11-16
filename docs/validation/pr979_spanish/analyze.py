import ast,hashlib,json,sqlite3,sys
from itertools import groupby
from pathlib import Path
import sqlalchemy
from sqlalchemy.orm import Session
from praatio import textgrid as pg
from montreal_forced_aligner import textgrid as mfa

R=Path(sys.argv[1])
def load_function(name):
 source=(R/name).read_text()
 node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='construct_textgrid_output')
 ns=dict(vars(mfa),groupby=groupby)
 exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),ns)
 return ns['construct_textgrid_output']
old=load_function('base_textgrid.py');new=load_function('pr_textgrid.py')
def parse(path):
 t=pg.openTextgrid(str(path),includeEmptyIntervals=False,reportingMode='error')
 return {n:[list(e) for e in t.getTier(n).entries] for n in t.tierNames}
def collection(folder):return {p.stem:parse(p) for p in folder.rglob('*.TextGrid')}
def kind(tiers,pattern):
 # Multi-speaker TextGrids use a suffix; single-speaker files use the tier name alone.
 return [e for n,entries in tiers.items() if n==pattern or n.endswith(' - '+pattern) for e in entries]
allruns={name:collection(R/'runs'/name/'textgrids') for name in ['original_base','original_pr','corrected_base','corrected_pr']}
comparisons={t:allruns[t+'_base']==allruns[t+'_pr'] for t in ['original','corrected']}
inputs=json.loads((R/'inputs.json').read_text())
samples=[]
for row in inputs:
 item={k:row[k] for k in ['id','duration','original','corrected','audio_sha256']}
 item['outputs']={name:dict(words=kind(data[row['id']],'words'),phones=kind(data[row['id']],'phones')) for name,data in allruns.items()}
 samples.append(item)

db=next((R/'runs/original_base').rglob('*.db'))
sha=hashlib.sha256(db.read_bytes()).hexdigest()
source=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
memory=sqlite3.connect(':memory:');source.backup(memory)
records=memory.execute('select f.id,f.name,f.relative_path,s.duration,t.text_file_path from file f join sound_file s on s.file_id=f.id join text_file t on t.file_id=f.id order by f.id').fetchall()
batch={i:(n,Path(rel or '.'),dur,Path(txt)) for i,n,rel,dur,txt in records}
target=records[0][0]
removed=memory.execute('select count(*) from phone_interval where utterance_id in(select id from utterance where file_id=?)',(target,)).fetchone()[0]
memory.execute('delete from phone_interval where utterance_id in(select id from utterance where file_id=?)',(target,));memory.commit()
engine=sqlalchemy.create_engine('sqlite://',creator=lambda:memory,poolclass=sqlalchemy.pool.StaticPool)
controlled={}
errors={}
for label,fn in [('base',old),('pr',new)]:
 dest=R/'controlled'/label;dest.mkdir(parents=True,exist_ok=True)
 try:
  with Session(engine) as session:list(fn(session,batch,True,"'",dest,.01,'long_textgrid',True))
 except KeyError as error:
  errors[label]=dict(type=type(error).__name__,message=str(error))
 controlled[label]=collection(dest)
target_name=records[0][1]
assert errors.get('base',{}).get('type')=='KeyError'
assert 'pr' not in errors
assert len(controlled['base'])==0
assert len(controlled['pr'])==2
for key in controlled['pr']:
 assert kind(controlled['pr'][key],'words')==kind(allruns['original_base'][key],'words')
assert kind(controlled['pr'][target_name],'phones')==[]
assert controlled['base']!=controlled['pr']
assert sha==hashlib.sha256(db.read_bytes()).hexdigest()
summary=dict(date='2026-10-27',base_commit='93411fc0df3bc770dfd34c4282dc136fdf6e8dcb',pr_commit='8545fd38f47b81f8a1743c8c9f8ab7b323e853eb',export_equal_without_injection=comparisons,samples=samples,
 controlled=dict(condition='All phone rows of the first file removed in an in-memory copy of the real original-text alignment database. This is deliberately injected, not naturally observed.',removed_phone_rows=removed,file=target_name,source_db_sha256=sha,source_db_unchanged=True,errors=errors,before=controlled['base'],after=controlled['pr']))
(R/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(dict(equal=comparisons,controlled_removed=removed,controlled_before=len(controlled['base']),controlled_after=len(controlled['pr']))))
