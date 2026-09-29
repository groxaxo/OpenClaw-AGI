"""Read-only cold reproduction of a fixed new-task experiment; never train."""
import argparse,json
from pathlib import Path
from .external_judge import ExternalJudgeGate,canonical,digest
from .new_task_runtime import Runtime,counts
from .new_task_suite import as_task
from .qwen4b_chain import file_sha256


def main():
    p=argparse.ArgumentParser()
    for key in ('run','model-path','start-record','public-suite','reserved-suite','registry','output'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args();run=Path(a.run);report=json.loads((run/'report.json').read_text());m=json.loads((run/'manifest.json').read_text())
    repo=Path(__file__).resolve().parents[2];gate=ExternalJudgeGate(run/'cold-attestation',repo,providers=('muse',))
    if digest(gate.sources())!=m['source_sha256']:raise RuntimeError('cold source mismatch')
    registry=json.loads(Path(a.registry).read_text())
    if file_sha256(a.reserved_suite)!=registry['reserved_sha256'] or file_sha256(a.public_suite)!=registry['public_sha256']:raise RuntimeError('suite mismatch')
    lock=json.loads((run/'candidate-locked.json').read_text());consumed=json.loads(Path(a.reserved_suite+'.consumed.json').read_text())
    if lock!=consumed or lock['candidate_sha256']!=report['candidate_sha256']:raise RuntimeError('candidate lock mismatch')
    final=json.loads(Path(a.reserved_suite).read_text());public=json.loads(Path(a.public_suite).read_text());start=json.loads(Path(a.start_record).read_text())
    rt=Runtime(a.model_path,m['max_new_tokens'],m['docker_image']);output={'status':'PASS','new_training_steps':0,'source_sha256':m['source_sha256'],'results':{}}
    checked=0
    for label,path,sha in [('saved_start',start['adapter_path'],start['adapter_tensor_state_sha256']),('selected_child',run/'selected',report['candidate_sha256'])]:
        rt.load(path,sha);output['results'][label]={}
        for kind in ('confirmation','transfer'):
            got=rt.evaluate([as_task(x) for x in final[kind]],'COLD_'+label+'_'+kind)
            expected=json.loads((run/(label+'-'+kind+'.json')).read_text());wanted={x['task_id']:x for x in expected}
            for row in got:
                prior=wanted[row['task_id']]
                if row['response_sha256']!=prior['response_sha256'] or row['result']!=prior['result']:raise RuntimeError('cold output mismatch '+row['task_id'])
                checked+=1
            output['results'][label][kind]=counts(got)
    guards=rt.evaluate([as_task(x) for x in public['regression']],'COLD_REGRESSION');expected=json.loads((run/'final-regression.json').read_text())
    for arow,brow in zip(guards,expected):
        if arow['task_id']!=brow['task_id'] or arow['response_sha256']!=brow['response_sha256'] or arow['result']!=brow['result']:raise RuntimeError('guard cold mismatch')
        checked+=1
    output.update(output_hashes_checked=checked,all_output_hashes_identical=True,regression=counts(guards),new_promotions=0)
    with Path(a.output).open('x') as f:f.write(canonical(output)+'\n')
    print(canonical(output),flush=True)

if __name__=='__main__':main()
