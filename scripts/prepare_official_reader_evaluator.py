"""Preparation-only opt-in source transformation; never imports/runs evaluator."""
import ast

LOAD = '        network, dual_identity = load_dual_fno('
LOAD_PRECISION = '''        # Historical checkpoint precision identity is validated at load.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
'''+LOAD
EVAL = '    network.eval()'
EVAL_PRECISION = EVAL+'''
    # Explicit confirmation-only post-load override, not original H100 protocol.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def confirmation_precision():
        flags = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                 "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
        if flags != {"float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}:
            raise RuntimeError("confirmation precision changed")
        return flags
    confirmation_precision()
'''
REPORT = '    report = {'
REPORT_PRECISION = '''    metadata["confirmation_inference_precision_override"] = confirmation_precision()
    metadata["confirmation_allocator_cap_bytes"] = min(float(cfg.training.gpu_memory_fraction) * torch.cuda.get_device_properties(dist.device).total_memory, 6 * 2**30) if dist.cuda else None
    if dist.cuda and metadata["confirmation_allocator_cap_bytes"] > 6 * 2**30:
        raise RuntimeError("allocator cap exceeds approved bound")
'''+REPORT
STEP = '                for offset in range(horizon):'
STEP_PRECISION = STEP+'\n                    confirmation_precision()'
CAP = '            float(cfg.training.gpu_memory_fraction), device=dist.device'
CAP_BOUNDED = '            min(float(cfg.training.gpu_memory_fraction), 6 * 2**30 / torch.cuda.get_device_properties(dist.device).total_memory), device=dist.device'
SPLIT = 'choices=("validation", "test"), default="test"'
SEALED_SPLIT = 'choices=("validation", "test", "frozen_test"), default="test"'
RECORD = '                        segment_records.append(segment_record)'
RECORD_FORCES = '''                        segment_record["predicted_force_channels"] = {channel: float(predicted_force[local_index, i]) for i, channel in enumerate(force_channels)}
                        segment_record["target_force_channels"] = {channel: float(selected_force[target_indices[local_index], i]) for i, channel in enumerate(force_channels)}
'''+RECORD

OLD = '''        with h5py.File(path, "r") as handle:
            state = torch.from_numpy(handle["state"][:]).to(dist.device)
            mask = torch.from_numpy(handle["mask"][:]).float().to(dist.device)
            omega = torch.from_numpy(handle["omega"][:]).float().to(dist.device)
            selected_force = (
                torch.from_numpy(handle["force"][:, list(force_indices)])
                .float()
                .to(dist.device)
            )
            time = torch.from_numpy(handle["time"][:]).float().to(dist.device)
            x = np.asarray(handle["x"][:])
            y = np.asarray(handle["y"][:])'''
NEW = '''        from official_evaluation_input import read_trajectory
        fields, coordinates = read_trajectory(path)
        state = fields["state"].to(dist.device)
        mask = fields["mask"].float().to(dist.device)
        omega = fields["omega"].float().to(dist.device)
        selected_force = fields["force"][:, list(force_indices)].float().to(dist.device)
        time = fields["time"].float().to(dist.device)
        x = coordinates["x"]
        y = coordinates["y"]'''

CHANGES = ((OLD,NEW),(LOAD,LOAD_PRECISION),(EVAL,EVAL_PRECISION),
           (REPORT,REPORT_PRECISION),(STEP,STEP_PRECISION),(CAP,CAP_BOUNDED),(SPLIT,SEALED_SPLIT),(RECORD,RECORD_FORCES))

def restore_source(source):
    for old,new in reversed(CHANGES):source=source.replace(new,old)
    return source

def opt_in_source(source):
    """Explicit input, post-load precision, and pre-load allocator-cap deltas."""
    result=source
    for old,new in CHANGES:
        if result.count(old)!=1:raise ValueError('reviewed source anchor drift')
        result=result.replace(old,new)
    ast.parse(result)
    if restore_source(result) != source:
        raise ValueError('unreviewed code changed')
    return result
