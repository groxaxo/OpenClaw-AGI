"""One-GPU before/after adaptation on new software tasks from a fixed artifact.

Public TRAIN/DEV can guide target validation and checkpoint selection. Reserved
new-input confirmation and transfer-only tasks are used once after selection.
No accepted checkpoint, global lineage or production service is changed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, threading, time
from pathlib import Path
from .coding_tasks import CodingTask, grade
from .external_judge import ExternalJudgeGate, JudgeError, canonical, digest
from .new_task_suite import as_task
from .new_task_runtime import Runtime, counts, paired, preserved, state_hash
from .qwen4b_chain import file_sha256, validate_parent
from .qwen4b_protected_repair import projected_delta
from .qwen4b_token_margins import choose_margins


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model-path','start-record','approved-record','public-suite','reserved-suite','registry','output','docker-image','test-report'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--max-epochs',type=int,default=3);parser.add_argument('--learning-rate',type=float,default=5e-5)
    parser.add_argument('--max-new-tokens',type=int,default=512);parser.add_argument('--max-wall-seconds',type=int,default=2700)
    a=parser.parse_args()
    if not 1<=a.max_epochs<=3 or not 1e-6<=a.learning_rate<=1e-4 or a.max_new_tokens!=512: parser.error('fixed bounded protocol violated')
    if not 300<=a.max_wall_seconds<=3000: parser.error('invalid wall budget')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700)
    repo=Path(__file__).resolve().parents[2]; started=time.monotonic(); finished=threading.Event()
    def write(name,value):
        with (out/name).open('x') as f:f.write(canonical(value)+'\n')
    def event(stage,**values):
        row={'stage':stage,'elapsed_seconds':round(time.monotonic()-started,3),**values}
        with (out/'events.jsonl').open('a') as f:f.write(canonical(row)+'\n')
        print(canonical(row),flush=True)
    def budget_watch():
        while not finished.wait(1):
            if time.monotonic()-started>a.max_wall_seconds or shutil.disk_usage(out).free<4*2**30:
                write('budget-failure.json',{'status':'NOT_OK','reason':'wall/disk budget'});os._exit(124)
    threading.Thread(target=budget_watch,daemon=True).start()
    event('started',physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    reg=json.loads(Path(a.registry).read_text())
    if file_sha256(a.public_suite)!=reg['public_sha256'] or file_sha256(a.reserved_suite)!=reg['reserved_sha256']:
        raise RuntimeError('suite preregistration hash mismatch')
    reserved=Path(a.reserved_suite); consumed=reserved.with_suffix(reserved.suffix+'.consumed.json')
    if consumed.exists():raise RuntimeError('reserved evaluation already consumed')
    public=json.loads(Path(a.public_suite).read_text())
    train,dev=[as_task(r) for r in public['train']],[as_task(r) for r in public['dev']]
    old_tasks=[as_task(r) for r in public['regression']]
    start_record=json.loads(Path(a.start_record).read_text()); approved=validate_parent(a.approved_record)
    if file_sha256(Path(start_record['adapter_path'])/'adapter_model.safetensors')!=start_record['adapter_file_sha256']:
        raise RuntimeError('saved starting artifact changed')
    judge=ExternalJudgeGate(out/'judges',repo,providers=('muse',),timeout_s=300)
    code_hash=digest(judge.sources()); test_report=json.loads(Path(a.test_report).read_text())
    if test_report.get('passed') is not True or test_report.get('code_sha256')!=code_hash:raise RuntimeError('stale tests')
    manifest={'backend':'new_task_bf16_token_margin_sft','starting_checkpoint_status':'DEVELOPMENT_ONLY cold-verified artifact',
        'comparator':'exact saved development checkpoint; approved generation 1 is a separate zero-shot reference',
        'starting_tensor_sha256':start_record['adapter_tensor_state_sha256'],'approved_tensor_sha256':approved['adapter_tensor_state_sha256'],
        'model':'Qwen/Qwen3.5-4B','revision':Path(a.model_path).name,'source_sha256':code_hash,'tests':test_report,
        'public_suite_sha256':reg['public_sha256'],'reserved_suite_sha256':reg['reserved_sha256'],
        'adaptation_families':10,'dev_cases':240,'confirmation_cases':240,'transfer_only_families':6,'transfer_cases':144,
        'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'max_optimizer_proposals':a.max_epochs,'learning_rate':a.learning_rate,
        'max_new_tokens':a.max_new_tokens,'max_wall_seconds':a.max_wall_seconds,'docker_image':a.docker_image,
        'determinism':'strict torch deterministic algorithms, cudnn deterministic, TF32 off, CUBLAS :4096:8; not by itself a reproducibility proof',
        'method':'verified failed-new-task reference SFT; three weakest token-margin anchors per solved public answer; full/half/quarter projected line search',
        'selection':'public DEV only, preserve every public old regression score and every solved new DEV task; final and transfer remain untouched',
        'experimental_acceptance_gate':{'min_case_gain':.08,'min_accuracy':.75,'min_new_complete':2,'max_lost_complete':0,'no_transfer_or_regression_case_loss':True},
        'scope':'new-to-this-experiment task families; adaptation confirmation has known prompts/new inputs; transfer tasks are excluded from adaptation',
        'production_deployed':False,'global_lineage_advanced':False}
    write('manifest.json',manifest)
    evidence={'manifest':manifest,'proposed_action':'bounded single-GPU new-task measurement; not production deployment or original-chain promotion'}
    pre=judge.review('preflight',evidence,deterministic_ok=True)
    if not pre.approved:raise JudgeError('preflight veto')
    judge.consume(pre,'preflight',evidence)
    rt=Runtime(a.model_path,a.max_new_tokens,a.docker_image)
    import torch
    import torch.nn.functional as F
    rt.load(start_record['adapter_path'],start_record['adapter_tensor_state_sha256']); starting=rt.snapshot()
    event('saved_checkpoint_loaded',sha256=state_hash(starting))
    write('runtime.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'trainable_parameters':sum(p.numel() for p in rt.params),
                         'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()})
    baseline=rt.evaluate(dev,'NEW_DEV0',out/'dev-baseline.json'); regress0=rt.evaluate(old_tasks,'REGRESSION0',out/'regression-baseline.json')
    incumbent_dev,incumbent_old=baseline,regress0; incumbent=starting; best=starting;best_score=counts(baseline)['case_accuracy'];best_epoch=0
    by_family={r['family']:r for r in public['train']}; repair=[]
    for task,record in zip(dev,baseline):
        family=task.task_id.removeprefix('dev_');row=by_family[family]
        verify= CodingTask('verify_'+family,row['description'],tuple((x,y) for x,y in row['cases'])+task.cases)
        old=grade(verify,record['response'],a.docker_image)
        if old['passed']==old['total']:continue
        result=grade(verify,row['solution'],a.docker_image)
        if result['passed']!=result['total']:raise RuntimeError('unverified supervised target')
        repair.append({'task_id':task.task_id,'description':task.description,'solution':row['solution'],'verified_cases':result['total']})
    write('training-targets.json',repair)
    event('targets_fixed',repair_count=len(repair),target_sha256=digest(repair))
    opt=torch.optim.AdamW(rt.params,lr=a.learning_rate,weight_decay=0.0)
    def flat_values():return torch.cat([p.detach().reshape(-1) for p in rt.params])
    def flat_grad():return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1) for p in rt.params]).clone()
    def copy_flat(x):
        at=0
        with torch.no_grad():
            for p in rt.params:p.copy_(x[at:at+p.numel()].view_as(p));at+=p.numel()
        if at!=x.numel():raise RuntimeError('adapter vector mismatch')
    def repair_loss(row):
        pids=rt.prompt_ids(row['description']);aids=rt.tokenizer(row['solution']+rt.tokenizer.eos_token,add_special_tokens=False)['input_ids']
        if len(pids)+len(aids)>1024:raise RuntimeError('training context budget')
        ids=torch.tensor([pids+aids],device='cuda:0');n=len(aids)
        logits=rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n+1).logits[:,:-1]
        if logits.shape[1]!=n:raise RuntimeError('response loss alignment')
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),ids[:,-n:].reshape(-1))
    history=[]
    for epoch in range(1,a.max_epochs+1):
        if not repair:break
        event('training_proposal',epoch=epoch);rt.model.train();opt.zero_grad(set_to_none=True)
        base_flat=flat_values();vectors=[];anchor_rows=[]
        for row in incumbent_dev+incumbent_old:
            if row['result']['passed']!=row['result']['total']:continue
            aids=row['response_token_ids'];n=len(aids);pids=row['prompt_token_ids']
            ids=torch.tensor([pids+aids],device='cuda:0')
            with torch.no_grad():
                logits=rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n+1).logits[0,:-1]
                anchors=choose_margins(logits,torch.tensor(aids,device='cuda:0'),count=3,counterexamples=[])
            for anchor in anchors:
                if len(vectors)>=96:raise RuntimeError('anchor budget exceeded')
                opt.zero_grad(set_to_none=True)
                prefix=torch.tensor([pids+aids[:anchor['position']]],device='cuda:0')
                values=rt.model(input_ids=prefix,attention_mask=torch.ones_like(prefix),use_cache=False,logits_to_keep=1).logits[0,-1].float()
                (values[anchor['chosen_token']]-values[anchor['competitor_token']]).backward()
                vectors.append(flat_grad());anchor_rows.append({'task_id':row['task_id'],**anchor})
        matrix=torch.stack(vectors) if vectors else torch.empty((0,base_flat.numel()),device='cuda:0')
        del vectors;opt.zero_grad(set_to_none=True);losses=[]
        for row in repair:
            loss=repair_loss(row)/len(repair);loss.backward();losses.append(float(loss.detach())*len(repair))
        norm=torch.nn.utils.clip_grad_norm_(rt.params,1.0,error_if_nonfinite=True)
        evidence={'manifest':manifest,'epoch':epoch,'mean_loss':sum(losses)/len(losses),'gradient_norm':float(norm),
            'actual_learning_rate':opt.param_groups[0]['lr'],'public_dev':counts(incumbent_dev),'public_regression':counts(incumbent_old),
            'dataset_sha256':digest(repair),'anchors_sha256':digest(anchor_rows),'anchor_count':len(anchor_rows),'adapter_before':state_hash(rt.snapshot()),
            'proposed_action':'one new-task SFT update, token-margin projection and public-only rollback search; no reserved/transfer result available'}
        ok=all(math.isfinite(x) for x in losses) and math.isfinite(float(norm)) and float(norm)>0
        approval=judge.review('before_training',evidence,deterministic_ok=ok)
        if not approval.approved:raise JudgeError('training veto')
        judge.consume(approval,'before_training',evidence);opt.step();opt.zero_grad(set_to_none=True)
        delta,projection=projected_delta(flat_values()-base_flat,matrix);del matrix
        copy_flat(base_flat+delta);proposal=rt.snapshot();del delta,base_flat
        trials=[];accepted=False;scale_accepted=0.0
        for scale in (1.0,.5,.25):
            blended={k:incumbent[k]+scale*(proposal[k]-incumbent[k]) for k in incumbent};rt.restore(blended)
            trial=rt.evaluate(dev,f'DEV{epoch}_{scale}',out/f'dev-{epoch}-{scale}.json')
            score=counts(trial); entry={'scale':scale,'dev':score,'public_dev_preserved':preserved(incumbent_dev,trial)}
            if entry['public_dev_preserved']:
                old=rt.evaluate(old_tasks,f'REGRESSION{epoch}_{scale}',out/f'regression-{epoch}-{scale}.json')
                entry['regression_preserved']=preserved(regress0,old)
                if entry['regression_preserved']:
                    incumbent=rt.snapshot();incumbent_dev=trial;incumbent_old=old;accepted=True;scale_accepted=scale
            trials.append(entry)
            if accepted:break
        if not accepted:rt.restore(incumbent)
        lr=opt.param_groups[0]['lr']*(1.0 if accepted else .5)
        opt=torch.optim.AdamW(rt.params,lr=max(1e-6,lr),weight_decay=0.0)
        folder=out/f'epoch-{epoch}';folder.mkdir();rt.model.save_pretrained(folder,safe_serialization=True)
        expected=state_hash(rt.snapshot());rt.load(folder,expected)
        item={'epoch':epoch,'dev':counts(incumbent_dev),'regression':counts(incumbent_old),'accepted':accepted,'accepted_scale':scale_accepted,
            'review_id':approval.review_id,'adapter_sha256':expected,'checkpoint_roundtrip_exact':True,'projection':projection,'trials':trials}
        history.append(item);write(f'epoch-{epoch}.json',item)
        if counts(incumbent_dev)['case_accuracy']>best_score:
            best=rt.snapshot();best_score=counts(incumbent_dev)['case_accuracy'];best_epoch=epoch
        event('proposal_completed',epoch=epoch,dev=counts(incumbent_dev),accepted=accepted)
        if best_score>=1.0 and epoch>=2:break
    # Fix the candidate before either heldout input set is opened. No adaptation
    # continues after this point, including after a rejection.
    rt.restore(best); selected=state_hash(best); candidate_dir=out/'selected';candidate_dir.mkdir();rt.model.save_pretrained(candidate_dir,safe_serialization=True)
    lock={'candidate_sha256':selected,'starting_sha256':state_hash(starting),'source_sha256':code_hash,'best_epoch':best_epoch,'run':str(out)}
    write('candidate-locked.json',lock)
    with consumed.open('x') as f:f.write(canonical(lock)+'\n')
    if file_sha256(reserved)!=reg['reserved_sha256']:raise RuntimeError('reserved suite changed')
    final=json.loads(reserved.read_text());confirmation=[as_task(r) for r in final['confirmation']];transfer=[as_task(r) for r in final['transfer']]
    results={}
    for label,path,sha in [('approved_reference',approved['adapter_path'],approved['adapter_tensor_state_sha256']),
                           ('saved_start',start_record['adapter_path'],start_record['adapter_tensor_state_sha256']),
                           ('selected_child',str(candidate_dir),selected)]:
        rt.load(path,sha)
        results[label]={'confirmation':rt.evaluate(confirmation,label+'_CONFIRM',out/(label+'-confirmation.json')),
                        'transfer':rt.evaluate(transfer,label+'_TRANSFER',out/(label+'-transfer.json'))}
    rt.load(candidate_dir,selected); final_guards=rt.evaluate(old_tasks,'FINAL_REGRESSION',out/'final-regression.json')
    change=paired(results['saved_start']['confirmation'],results['selected_child']['confirmation'])
    transfer_change=paired(results['saved_start']['transfer'],results['selected_child']['transfer'])
    zero_shot=paired(results['approved_reference']['confirmation'],results['saved_start']['confirmation'])
    transfer_zero=paired(results['approved_reference']['transfer'],results['saved_start']['transfer'])
    deterministic_gate=bool(change['case_gain']>=.08 and change['after']['case_accuracy']>=.75 and change['newly_complete']>=2 and change['lost_complete']==0
        and preserved(results['saved_start']['transfer'],results['selected_child']['transfer']) and preserved(regress0,final_guards) and selected!=state_hash(starting))
    report={'execution_status':'COMPLETE','model':'Qwen/Qwen3.5-4B','starting_status':'DEVELOPMENT_ONLY','scope':manifest['scope'],
        'adaptation_confirmation':change,'transfer_only':transfer_change,'previous_adapter_zero_shot':zero_shot,'previous_adapter_transfer':transfer_zero,
        'dev_baseline':counts(baseline),'selected_dev_accuracy':best_score,'regression_before':counts(regress0),'regression_after':counts(final_guards),
        'candidate_sha256':selected,'starting_sha256':state_hash(starting),'best_epoch':best_epoch,'optimizer_proposals':len(history),'history':history,
        'deterministic_acceptance_passed':deterministic_gate,'global_lineage_advanced':False,'production_deployed':False,
        'memory_peak_allocated_gib':torch.cuda.max_memory_allocated(0)/2**30,'elapsed_before_final_reviews_s':time.monotonic()-started}
    evidence={'manifest':manifest,'report':report,'proposed_action':'accept new-task experimental artifact only; no global lineage or serving change'}
    decision=judge.review('checkpoint_promotion',evidence,deterministic_ok=deterministic_gate)
    report['experimental_acceptance_approved']=decision.approved;report['acceptance_review_id']=decision.review_id
    if decision.approved:judge.consume(decision,'checkpoint_promotion',evidence)
    final_evidence={'manifest':manifest,'report':report,'proposed_action':'audit completed measurement including any zero/negative result; NOT a performance or reproducible-training certification'}
    audit=judge.review('final_review',final_evidence,deterministic_ok=True)
    report['measurement_audit_approved']=audit.approved;report['final_review_id']=audit.review_id
    report['total_elapsed_seconds']=time.monotonic()-started;write('report.json',report);finished.set()
    print(canonical(report),flush=True)
    if not audit.approved:raise SystemExit(4)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import sys
        try:
            folder=Path(sys.argv[sys.argv.index('--output')+1])
            if folder.exists() and not (folder/'failure.json').exists():
                (folder/'failure.json').write_text(canonical({'status':'NOT_OK','type':type(exc).__name__,'error':str(exc)})+'\n')
        except (ValueError,IndexError,OSError):pass
        raise
