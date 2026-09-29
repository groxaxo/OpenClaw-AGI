"""Read-only audit of the completed new-task measurement; never trains."""
import json, re
from pathlib import Path
from dream_rsi.external_judge import ExternalJudgeGate, canonical, digest, extract_response, parse_verdict
from dream_rsi.new_task_runtime import counts, paired, preserved, state_hash
from dream_rsi.qwen4b_chain import validate_parent, file_sha256
from safetensors.torch import load_file
T=Path('/home/op/dream-qwen4b-newtasks.sIzsVr'); R=T/'run-002'
def read(path): return json.loads(path.read_text())
def require(condition, message):
    if not condition: raise RuntimeError(message)
m=read(R/'manifest.json'); report=read(R/'report.json')
require((T/'exit-code-002.txt').read_text().strip()=='0','measurement process did not finish')
require(report['execution_status']=='COMPLETE' and report['measurement_audit_approved'] is True,'no completed audit')
g=ExternalJudgeGate(T/'audit-attestation',T/'repo',providers=('muse',))
require(digest(g.sources())==m['source_sha256'],'source mismatch')
parent=validate_parent(T/'approved-parent.json'); start=read(T/'starting-candidate.json')
require(file_sha256(Path(start['adapter_path'])/'adapter_model.safetensors')==start['adapter_file_sha256'],'saved start changed')
require(parent['adapter_tensor_state_sha256']==m['approved_tensor_sha256'],'approved parent mismatch')
require(start['adapter_tensor_state_sha256']==m['starting_tensor_sha256'],'starting identity mismatch')
registry=read(T/'suites/registry.json')
for name,key in [('public.json','public_sha256'),('reserved.json','reserved_sha256')]:
    require(file_sha256(T/'suites'/name)==registry[key],'suite bytes changed')
lock=read(R/'candidate-locked.json')
require(lock==read(T/'suites/reserved.json.consumed.json'),'reserved candidate lock mismatch')
require(lock['candidate_sha256']==report['candidate_sha256'],'candidate changed after lock')
require(state_hash(load_file(R/'selected/adapter_model.safetensors'))==report['candidate_sha256'],'selected adapter mismatch')
receipts=[]; seen=set()
for folder in (R/'judges').iterdir():
    if not (folder/'request.json').exists(): continue
    req=read(folder/'request.json'); a=read(folder/'approval.json')
    require(digest(req)==a['review_id']==folder.name,'review request identity mismatch')
    require(a['code_sha256']==m['source_sha256'],'review source mismatch')
    require(a['evidence_sha256']==digest(req['evidence']),'review evidence mismatch')
    final=parse_verdict(extract_response('muse',(folder/'muse.jsonl').read_text(),'muse-spark-1.3'),folder.name)
    require(final['verdict']==a['verdicts']['muse']['verdict'],'terminal verdict mismatch')
    if a['stage']=='checkpoint_promotion':
        require(not a['approved'] and not req['deterministic_ok'],'failed promotion authorized')
        require(not (folder/'consumed.json').exists(),'rejected action consumed')
    else:
        require(a['approved'] and final['verdict']=='OK','required review rejected')
        if a['stage']!='final_review': require((folder/'consumed.json').exists(),'action approval unconsumed')
    seen.add(a['stage']); receipts.append(a)
require(len(receipts)==6 and seen=={'preflight','before_training','checkpoint_promotion','final_review'},'review count mismatch')
last=m['starting_tensor_sha256']
for row in report['history']:
    req=read(R/'judges'/row['review_id']/'request.json')
    require(req['evidence']['adapter_before']==last,'optimizer lineage mismatch')
    require(req['evidence']['epoch']==row['epoch'],'optimizer approval for wrong step')
    require(row['checkpoint_roundtrip_exact'] is True,'checkpoint roundtrip missing')
    require(state_hash(load_file(R/f"epoch-{row['epoch']}"/'adapter_model.safetensors'))==row['adapter_sha256'],'epoch bytes mismatch')
    last=row['adapter_sha256']
require(len(report['history'])==3,'unexpected optimizer count')
require(report['best_epoch']==lock['best_epoch']==3,'unrecorded candidate selection')
tables={label:{kind:read(R/f'{label}-{kind}.json') for kind in ('confirmation','transfer')}
        for label in ('approved_reference','saved_start','selected_child')}
adapt=paired(tables['saved_start']['confirmation'],tables['selected_child']['confirmation'])
transfer=paired(tables['saved_start']['transfer'],tables['selected_child']['transfer'])
require(adapt==report['adaptation_confirmation'],'confirmation arithmetic mismatch')
require(transfer==report['transfer_only'],'transfer arithmetic mismatch')
reg0=read(R/'regression-baseline.json'); reg1=read(R/'final-regression.json')
require(counts(reg0)==report['regression_before'] and counts(reg1)==report['regression_after'],'regression totals mismatch')
require(preserved(reg0,reg1),'public regression loss')
deterministic=bool(adapt['case_gain']>=.08 and adapt['after']['case_accuracy']>=.75
    and adapt['newly_complete']>=2 and adapt['lost_complete']==0
    and preserved(tables['saved_start']['transfer'],tables['selected_child']['transfer'])
    and preserved(reg0,reg1) and report['candidate_sha256']!=report['starting_sha256'])
require(deterministic==report['deterministic_acceptance_passed']==False,'acceptance gate incorrectly reported')
require(report['experimental_acceptance_approved'] is False,'candidate incorrectly accepted')
require(report['production_deployed'] is False and report['global_lineage_advanced'] is False,'unexpected deployment or promotion')
events=[json.loads(x) for x in (R/'events.jsonl').read_text().splitlines()]
require(sum(x['stage']=='proposal_completed' for x in events)==3,'incomplete proposals')
require((R/'candidate-locked.json').stat().st_mtime>(R/'epoch-3.json').stat().st_mtime,'candidate not fixed after selection')
for label in tables:
    for kind in ('confirmation','transfer'):
        require((R/f'{label}-{kind}.json').stat().st_mtime>=(R/'candidate-locked.json').stat().st_mtime,'evaluation before candidate lock')
unit_log=(T/'evidence/final-unit-tests.log').read_text()
require(re.search(r'Ran 150 tests',unit_log) and unit_log.rstrip().endswith('OK'),'unit evidence incomplete')
original=Path('/home/op/dream-qwen4b-chain.8nWOF3/sealed/generation2-confirmation.json')
require(not original.with_suffix(original.suffix+'.consumed.json').exists(),'original-chain confirmation consumed')
record={'status':'PASS','scope':'completed measurement and receipt integrity; NOT checkpoint acceptance',
 'model_acceptance':'NOT_OK','source_sha256':m['source_sha256'],'tests_run':150,
 'review_count':len(receipts),'optimizer_proposals':3,'accepted_public_updates':sum(x['accepted'] for x in report['history']),
 'confirmation':adapt,'transfer':transfer,'public_development_before':report['dev_baseline'],
 'public_development_after':report['history'][-1]['dev'],'public_regression_before':counts(reg0),'public_regression_after':counts(reg1),
 'zero_shot':report['previous_adapter_zero_shot'],'previous_adapter_transfer':report['previous_adapter_transfer'],
 'candidate_sha256':report['candidate_sha256'],'candidate_file_sha256':file_sha256(R/'selected/adapter_model.safetensors'),
 'approved_parent_unchanged':True,'saved_start_unchanged':True,'original_chain_confirmation_consumed':False,
 'new_task_reserved_suite_consumed':True,'new_promotions':0,'production_deployed':False}
with (T/'evidence/completed-measurement-audit.json').open('x') as f:f.write(canonical(record)+'\n')
with (T/'evidence/completed-review-receipts.json').open('x') as f:f.write(canonical(receipts)+'\n')
print(canonical({k:v for k,v in record.items() if k not in {'confirmation','transfer','zero_shot','previous_adapter_transfer'}}),flush=True)
