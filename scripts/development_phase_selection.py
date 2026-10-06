"""Fixed, already-opened b01/b03 controlled development replay selection.

Selection only: no model, CFD, reader or filesystem writes.  Its records use
the reviewed R2 converter/core schema without altering those implementations.
"""
from pathlib import Path
import math

PHASE_START = {'b01': 130.0, 'b03': 144.0}
STARTS = tuple(range(0, 800, 100))


def require(value, message):
    if not value:
        raise ValueError(message)


def observation(row, key):
    value = row.get(key)
    require(isinstance(value, list) and len(value) == 69
            and all(type(x) in (int, float) and math.isfinite(x) for x in value), key)
    return value


def make_selection(document, source_root, phase, *, require_files=True):
    require(phase in PHASE_START, 'fixed development phase')
    start_time = PHASE_START[phase]
    rows = document.get('rows')
    require(type(document.get('completed_cycles')) is int
            and document['completed_cycles'] == 800
            and isinstance(rows, list) and len(rows) == 800, 'completed800')
    previous = 0.0
    for i, row in enumerate(rows):
        require(type(row.get('step')) is int and row['step'] == i+1, 'step')
        for key in ('start_time', 'end_time', 'applied_omega', 'requested_omega'):
            require(type(row.get(key)) in (int, float) and math.isfinite(row[key]), key)
        require(abs(row['start_time']-(start_time+.1*i)) <= 1e-8
                and abs(row['end_time']-(start_time+.1*(i+1))) <= 1e-8, 'phase clock')
        applied = row['applied_omega']
        require(abs(applied) <= .75+1e-12 and abs(applied-previous) <= .1+1e-12
                and abs(row['requested_omega']) <= .75+1e-12, 'action limits')
        before = observation(row, 'input_observation')
        after = observation(row, 'output_observation')
        require(abs(before[68]-previous) <= 1e-7
                and abs(after[68]-applied) <= 1e-7, 'action clock')
        previous = applied
    source_root = Path(source_root)
    records = []
    for start in STARTS:
        frames = []
        for offset in range(6):
            index = start+offset
            stamp = start_time+.1*index
            relative = Path('case_mpc')/f'{stamp:.10g}'
            paths = {key: relative/key for key in ('U', 'p')}
            if require_files:
                require(all((source_root/path).is_file()
                            and not (source_root/path).is_symlink() for path in paths.values()),
                        'saved controlled field')
            frames.append(dict(global_index=index, time=stamp, source=str(relative),
                               files={key:str(path) for key,path in paths.items()}))
        transitions = []
        for offset in range(5):
            i = start+offset
            transitions.append(dict(lead=offset+1,
                omega_now=0.0 if i == 0 else rows[i-1]['applied_omega'],
                omega_next=rows[i]['applied_omega'],
                truth_force=observation(rows[i], 'output_observation')[64:68]))
        records.append(dict(branch='mpc', phase=phase, start_index=start,
            start_time=frames[0]['time'], frames=frames,
            initial_force=observation(rows[start], 'input_observation')[64:68],
            transitions=transitions))
    return dict(status='DEVELOPMENT_PHASE_SELECTION_PREPARATION_ONLY', phase=phase,
        split='development_already_opened', starts=list(STARTS), horizon=5,
        frames=48, endpoints=40, records=records, scientific_admission=False)
