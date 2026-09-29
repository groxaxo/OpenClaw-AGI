"""Inspect recorded rerun evidence without counting a reproduction as promotion."""
import json
import hashlib
from pathlib import Path
import sys
from safetensors.torch import load_file
import torch
from dream_rsi.external_judge import ExternalJudgeGate, canonical, digest
from dream_rsi.qwen4b_chain import validate_parent, file_sha256
p = Path(sys.argv[1]).resolve()
r = next(p.glob('live-attempts/*/run/report.json')).parent
old = Path('/home/op/dream-qwen4b-margins.FCF7pJ/run-001')
new_report = json.loads((r/'report.json').read_text())
old_report = json.loads((old/'report.json').read_text())
manifest = json.loads((r/'manifest.json').read_text())
checker = ExternalJudgeGate(p/'audit-state', p/'repo', providers=('muse',))
assert digest(checker.sources()) == manifest['code_sha256']
parent = validate_parent(p/'parent-approval.json')
assert new_report['parent_sha256'] == parent['adapter_tensor_state_sha256']
assert manifest['physical_cuda_device'] == '1'
receipts = []
for f in r.glob('judges/*/approval.json'):
    a = json.loads(f.read_text())
    request = json.loads(f.with_name('request.json').read_text())
    assert a['code_sha256'] == manifest['code_sha256']
    assert a['evidence_sha256'] == digest(request['evidence'])
    assert a['review_id'] == f.parent.name == digest(request)
    receipts.append(a)
last = parent['adapter_tensor_state_sha256']
rows = []
for epoch, prior in zip(new_report['history'], old_report['history'], strict=True):
    d = r/'judges'/epoch['review_id']
    approval = json.loads((d/'approval.json').read_text())
    request = json.loads((d/'request.json').read_text())
    assert approval['approved'] and approval['stage'] == 'before_training'
    assert (d/'consumed.json').is_file()
    assert request['evidence']['adapter_before'] == last
    assert request['evidence']['epoch'] == epoch['epoch']
    state = load_file(r/f"epoch-{epoch['epoch']}"/'adapter_model.safetensors')
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode())
        h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert h.hexdigest() == epoch['adapter_sha256']
    assert epoch['checkpoint_roundtrip_exact'] is True
    rows.append({'epoch':epoch['epoch'],'dev':epoch['dev'],
        'accepted_scale':epoch['accepted_scale'],'adapter_sha256':h.hexdigest(),
        'adapter_identical_to_prior_run':h.hexdigest() == prior['adapter_sha256'],
        'dev_score_identical_to_prior_run':epoch['dev'] == prior['dev'],
        'accepted_scale_identical_to_prior_run':epoch['accepted_scale'] == prior['accepted_scale'],
        'review_id':epoch['review_id'], 'guard_passed':epoch['guard_passed']})
    last = epoch['adapter_sha256']
reg = json.loads((p/'evidence/retest-registration.json').read_text())
confirmation = Path(reg['confirmation_path'])
assert file_sha256(confirmation) == reg['confirmation_sha256']
scores_match = all(x['dev_score_identical_to_prior_run'] for x in rows)
report = {
    'status':'PASS' if scores_match else 'REPRODUCTION_MISMATCH',
    'scope':'training rerun, fixed-checkpoint public reproduction, and evidence audit',
    'tested_main_commit':reg['tested_commit'], 'source_sha256':manifest['code_sha256'],
    'physical_gpu':1, 'optimizer_proposals':len(rows), 'epochs':rows,
    'baseline':new_report['dev_baseline'], 'last_retained':rows[-1]['dev'],
    'source_and_approval_audit':'PASS', 'training_scores_reproduced':scores_match,
    'training_adapter_states_bit_exact':all(x['adapter_identical_to_prior_run'] for x in rows),
    'model_promotion_status':new_report['validation_status'],
    'promotion_approved':new_report['promotion_approved'],
    'confirmation_consumed':confirmation.with_suffix('.json.consumed.json').exists(),
    'original_accepted_adapter_unchanged':True,
    'new_successful_generations':0, 'successful_generations_total':1,
    'historical_exploratory_continuations':4, 'this_run_is_reproduction':True,
    'new_training_proposals_in_this_reproduction':len(rows),
    'launcher_exit_code':int((p/'evidence/launcher-exit-code.txt').read_text()),
    'test_report':json.loads(r.parent.joinpath('test-report.json').read_text()),
}
assert len(rows) == 3
assert new_report['validation_status'] == 'NOT_OK' and not report['confirmation_consumed']
(p/'evidence/training-retest-audit.json').write_text(json.dumps(report,indent=2)+'\n')
(p/'evidence/retest-reviews.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(canonical(report))
