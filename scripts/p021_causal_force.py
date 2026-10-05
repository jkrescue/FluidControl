"""Project-owned FC-P021 CPU engineering adapters; not a candidate/trainer."""
from pathlib import Path
import hashlib
import json
import importlib.util
import torch
from torch.utils.checkpoint import checkpoint

AUDIT_SHA='72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2'
NORM_SHA='f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
OFFICIAL_SHA='e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9'
OBJECTIVE_SHA='f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7'
LIFT='spec_encoder.lift_network.0.conv.weight'
FROZEN=('spec_encoder.lift_network.0.conv.bias','spec_encoder.lift_network.2.conv.bias')
IDENTITIES=((160,'matched_start_acquisition_train_b00_zero',320),(816,'dynamic_train8_b00_prbs',90),
 (923,'dynamic_train8_b02_prbs',100),(975,'dynamic_train8_b04_prbs',0),
 (1077,'dynamic_train8_b06_prbs',0),(1233,'direct_cfd_directppo2048_v1_env0_ep0009_b00',0))

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_objective(path):
    if sha(path)!=OBJECTIVE_SHA:raise ValueError('immutable objective SHA differs')
    spec=importlib.util.spec_from_file_location('p021_original_objective',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.balanced_force_objective

def normalize(force,mean,std):
    if force.shape[-1]!=4 or mean.shape!=(4,) or std.shape!=(4,) or not torch.isfinite(force).all() or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('finite four-force train normalization required')
    return (force-mean)/std

def load_causal_inputs(audit_path,normalization_path,repo):
    """Only reviewed six windows; rehash small raw sources, never scan HDF fields."""
    if sha(audit_path)!=AUDIT_SHA or sha(normalization_path)!=NORM_SHA:raise ValueError('pinned input proof differs')
    audit=json.loads(Path(audit_path).read_text());stats=json.loads(Path(normalization_path).read_text());repo=Path(repo).resolve()
    for name,digest in audit['raw_source_sha256'].items():
        path=(repo/name).resolve()
        if not path.is_relative_to(repo) or sha(path)!=digest:raise ValueError('raw source identity differs')
    mean=torch.tensor(stats['all_force_mean'],dtype=torch.float32);std=torch.tensor(stats['all_force_std'],dtype=torch.float32)
    if stats['all_force_channels']!=['front_cd','front_cl','rear_cd','rear_cl']:raise ValueError('force channel order differs')
    outputs=[]
    if len(audit['rows'])!=6:raise ValueError('six windows required')
    for row,(index,case,start) in zip(audit['rows'],IDENTITIES):
        identity=row['identity']
        if (row['global_index'],identity['case'],identity['start'])!=(index,case,start) or identity['split']!='train':raise ValueError('fixed train identity differs')
        force=torch.tensor(row['exact_current_physical_force'],dtype=torch.float32)
        if force.shape!=(101,4):raise ValueError('101 exact force rows required')
        for obj in row['objects']:
            if len(obj['frames'])!=101:raise ValueError('101 provenance rows required')
            for offset,frame in enumerate(obj['frames']):
                t=row['nominal_time_origin']+.1*(start+offset)
                if frame['frame']!=start+offset or abs(frame['canonical_time']-t)>1e-8 or abs(frame['exact_raw_time']-t)>1e-8 or frame['exact_source_file'] not in audit['raw_source_sha256']:raise ValueError('noncausal or unbound endpoint')
        outputs.append(dict(global_index=index,identity=identity,physical=force,normalized=normalize(force,mean,std),hdf_sha256=row['hdf_sha256_from_existing_audit']))
    return outputs,mean,std

def expanded_state(old,new_template):
    """Copy physical6 and appended coords2; zero only added force columns."""
    if old.keys()!=new_template.keys():raise ValueError('model state keys differ')
    out={}
    for name,value in old.items():
        target=new_template[name]
        if value.dtype!=target.dtype:raise ValueError('model dtype differs')
        if name==LIFT:
            if value.ndim!=4 or value.shape[1]!=8 or target.shape[1]!=12 or value.shape[0]!=target.shape[0] or value.shape[2:]!=target.shape[2:]:raise ValueError('official lifting shape differs')
            result=torch.zeros_like(target);result[:,:6]=value[:,:6];result[:,10:12]=value[:,6:8];out[name]=result
        else:
            if value.shape!=target.shape:raise ValueError('nonlifting shape differs')
            out[name]=value.detach().clone()
    return out

def warmstart(old,new,official_source):
    if sha(official_source)!=OFFICIAL_SHA:raise ValueError('official encoder source differs')
    state=expanded_state(old.state_dict(),new.state_dict());new.load_state_dict(state,strict=True)
    names=dict(new.named_parameters())
    for name in FROZEN:
        if name not in names:raise ValueError('missing frozen lifting bias')
        names[name].requires_grad_(False)
    for name,param in names.items():
        if name not in FROZEN:param.requires_grad_(True)
    if sum(p.requires_grad for p in names.values())!=28:raise ValueError('expected original28 trainable tensors')
    return new

def validate_paired_model(model):
    """Reject known stateful/stochastic layers; not certification of custom code.

    The eventual official model must additionally match its pinned source/schema.
    Batch statistics could couple teacher-forced H1 inputs into the AR branch;
    mutable buffers or stochastic forwards also invalidate checkpoint replay.
    """
    for layer in model.modules():
        if isinstance(layer,(torch.nn.modules.batchnorm._BatchNorm,
                             torch.nn.modules.dropout._DropoutNd,
                             torch.nn.RNNBase)):
            raise ValueError('paired recurrence forbids batch statistics/dropout/stateful recurrence')
        if isinstance(layer,torch.nn.modules.instancenorm._InstanceNorm) and layer.track_running_stats:
            raise ValueError('paired recurrence forbids mutable running statistics')
    for name,buffer in model.named_buffers():
        # Observed official Module device markers contain no state values.
        if name.split('.')[-1]!='device_buffer' or buffer.shape!=(0,) or buffer.dtype!=torch.float32:
            raise ValueError('unreviewed model buffers could mutate during replay')

def recurrent_objective(model,h1_inputs,ar_inputs,mask,current_forces,targets,predict,balanced,*,zero_conditioning=False,checkpointed=True):
    """H100 full BPTT; checkpoint recomputes ten steps, never detaches force."""
    validate_paired_model(model)
    if h1_inputs.shape!=ar_inputs.shape or h1_inputs.shape[:2]!=(100,6) or current_forces.shape!=(101,4) or targets.shape!=(100,4):raise ValueError('fixed H100 four-force contract differs')
    if any(x.requires_grad for x in (h1_inputs,ar_inputs,mask,current_forces,targets)):raise ValueError('frozen data/flow must not require gradient')
    if any(not torch.isfinite(x).all() for x in (h1_inputs,ar_inputs,mask,current_forces,targets)):raise ValueError('nonfinite input')
    previous=current_forces[0];h1_parts=[];ar_parts=[]
    for begin in range(0,100,10):
        def block(previous,begin=begin):
            hp=[];ap=[]
            for k in range(begin,begin+10):
                condition=torch.stack((current_forces[k],previous))
                if zero_conditioning:condition=torch.zeros_like(condition)
                planes=condition[:,:,None,None].expand(-1,-1,*h1_inputs.shape[-2:])
                inputs=torch.cat((torch.stack((h1_inputs[k],ar_inputs[k])),planes),dim=1)
                _,force=predict(model,inputs,mask.expand(2,-1,-1,-1))
                if force.shape!=(2,4):raise ValueError('paired force outputs required')
                hp.append(force[0]);ap.append(force[1]);previous=force[1]
            return previous,torch.stack(hp),torch.stack(ap)
        previous,hp,ap=checkpoint(block,previous,use_reentrant=False,preserve_rng_state=True) if checkpointed else block(previous)
        h1_parts.append(hp);ar_parts.append(ap)
    h1=torch.cat(h1_parts);ar=torch.cat(ar_parts)
    if not torch.isfinite(h1).all() or not torch.isfinite(ar).all():raise FloatingPointError('nonfinite force recurrence')
    hl=balanced(h1[None],targets[None]);al=balanced(ar[None],targets[None])
    return dict(total=.5*(hl['balanced']+al['balanced']),h1=h1,ar=ar,h1_loss=hl,ar_loss=al)

def persistence(exact_current,unchanged_targets):
    """Physical strict-causal reference; no target is fed back in AR."""
    if exact_current.shape!=(101,4) or unchanged_targets.shape!=(100,4):raise ValueError('H100 shapes required')
    y=unchanged_targets.double();out={}
    for domain,p in [('h1',exact_current[:-1].double()),('ar',exact_current[:1].double().expand(100,4))]:
        r=p-y;t=r[38:];pr=p[38:].std(0,correction=0);yr=y[38:].std(0,correction=0)
        out[domain]=dict(mae=r.abs().mean(0).tolist(),bias_mse=t.mean(0).square().tolist(),centered_residual_mse=(t-t.mean(0)).square().mean(0).tolist(),absolute_rms_error=(pr-yr).abs().tolist(),predicted_rms=pr.tolist(),truth_rms=yr.tolist())
    return out
