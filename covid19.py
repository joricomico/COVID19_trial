# pylint: disable=no-self-argument, no-member

import sys
sys.path.append('..')

from core import REc, struct
from cop import load_data, translation, patient, model, idsets_of

from numpy import array, ones, average, median, std
from scipy import stats

def make_model(by_vars, data, groups):
    '''make model for testing'''
    T = translation(by_vars)
    pclass = patient(by_vars)
    pclass.load(data, T)
    return model(pclass, groups=groups)

def make_db_model(by_vars, data, groups):
    '''code to create new subjec repo'''

def _calculate_models_common_init(): return []
def _calculate_models_common_calc(hub, calc_args, tries, message, header='{} tries before saving...'):
    print('{} tries before test is ready...'.format(tries))
    print(message.format(hub.DB.ids, hub.DB.long_ids))
    hub.calculate(**calc_args)
    hub.best = hub.unstacked
    return struct(vars=hub.vars_of, stats=hub.stats)
def _calculate_models_common_redo(hub, result, stack, and_retry):
    hub.best, hub.unstacked = [], []
    stack.append(result)
    return and_retry-1

def calculate_models_from_end(hub, calc_args, message, retries):
    '''test service: calculate models from the end of the stay,
    ideal for good outcome prediction and early discharge.'''
    results = _calculate_models_common_init()
    while retries:
        result = _calculate_models_common_calc(hub, calc_args, retries, message)
        ir, dc = set([int(s.hn) for s in hub.in_room]), set([int(s.hn) for s in hub.dismissed])
        result(in_room=ir, discharged=dc)
        retries = _calculate_models_common_redo(hub, result, results, retries)
    return hub, results

def calculate_models_from_start(hub, calc_args, message, retries):
    '''test service: calculate models from the start of the stay,
    ideal for bad outcome prediction and worsening.'''
    results = _calculate_models_common_init()
    while retries:
        result = _calculate_models_common_calc(hub, calc_args, retries, message)
        assessed = set([int(s.hn) for s in hub.assessed])
        result(assessed=assessed)
        retries = _calculate_models_common_redo(hub, result, results, retries)
    return hub, results    

class FOR:
    EarlyDischarge = {(0,):0, range(1,5):1}
    EarlyDischargeO2 = {(0,1):0, range(2,5):1}
    Worsening = {(0,1):0, range(2,6):1}
    MechVent = {(0,1,2):0, range(3,6):1}

def test(varfile, csvdb, groups=FOR.EarlyDischarge, on=None, message='', thresh=.5, err0=.05, retry=2):
    '''creates a test model for a specific group defined by FOR, 
    returns a model and test results, to avoid testing use retry=0.'''
    model, calcargs, compute = make_model(varfile, csvdb, groups), None, calculate_models_from_start
    if groups == FOR.EarlyDischarge:
        if on is None: on = .7
        if message == '': message = 'anticipate deterioration or recovery: {} subjects, {} in room'
        calcargs = dict(on=on, virtual=int(model.DB.long_ids*.2), outcome_from=0, err=err0, from_end=True, threshold=thresh, skip=1, targets=[])
        compute = calculate_models_from_end
    elif groups == FOR.EarlyDischargeO2:
        if on is None: on = .7
        if message == '': message = 'anticipate deterioration or recovery from minimal O2 ventilation: {} subjects, {} in room'
        calcargs = dict(on=on, virtual=int(model.DB.long_ids*.2), outcome_from=1, err=err0, from_end=True, threshold=thresh, skip=1, targets=[])
        compute = calculate_models_from_end
    elif groups == FOR.Worsening:
        if on is None: on = .6
        if message == '': message = 'anticipate worsening: {} subjects, {} in room'
        calcargs = dict(on=on, virtual=int(model.DB.long_ids*.3), outcome_from=3, err=err0, threshold=thresh, skip=0)
    elif groups == FOR.MechVent:
        if on is None: on = .7
        if message == '': message = 'anticipate mechanical ventilation and ICU: {} subjects, {} in room'
        calcargs = dict(on=on, virtual=int(model.DB.long_ids/5), outcome_from=4, err=err0, threshold=thresh, skip=0)
    if not retry: model.set(err=err0)
    model.set(threshold=thresh)
    return compute(model, calcargs, message, retry)

def get_vars_from(results):
    vars = {l:[] for l,_ in results[0].vars}
    for data in results:
        for l,_set in data.vars:
            vars[l] += _set
    return vars

def get_stats_from(results):
    stat = {s:[] for s,_set in results[0].stats.items()}
    for data in results:
        for s,_set in data.stats.items():
            stat[s] += _set
    return stat

def get_subjects_from(results, force_room_home=False):
    room, home = [], []
    for data in results:
        room += list(data.in_room) if 'in_room' in data else list(data.assessed)
        home += list(data.discharged) if 'discharged' in data else []
    if not force_room_home and home == []: return room
    return room, home

def get_subject_number_from(results):
    room, home = get_subjects_from(results, True)
    return len(set(room+home))

def list_subjects_from(data):
    base = get_subjects_from(data)
    if type(base) is tuple: return list(set(base[0]+base[1]))
    return list(set(base))

def AUC_of(stats, target='before', comparator='good', _from=0, show=True):
    values = [v for source in stats for v in source.stats[target]]
    sens = 1-sum([int(v<_from) for v in values])/len(values)
    spec = average([s for source in stats for s in source.stats[comparator]])
    AUC = (sens+spec)/2
    if show:
        print('sensitivity: {:.1%}'.format(sens))
        print('specificity: {:.1%}'.format(spec))
        print('ROC-AUC:     {:.1%}'.format(AUC))
    return struct(sens=sens, spec=spec, AUC=AUC)

def show_(vars, force_avg=False, show=True):
    for label,_set in vars.items():
        normal = stats.shapiro(_set)[1] if len(_set)<5000 else stats.normaltest(_set)[1]
        value = average(_set) if normal>=0.05 or force_avg else median(_set)
        rng = std(_set) if normal>0.05 or force_avg else stats.iqr(_set)
        print('{}:\t{:.2f}±{:.3f} ({:.3f})'.format(label, value, rng, normal))
        results = struct(**{label:struct(value=value, range=rng, normal=normal)})
    return results

def organize_result(model, data, show=True):
    S = struct(vars=get_vars_from(data), stats=get_stats_from(data), real=list_subjects_from(data), virtual=get_subject_number_from(data))
    auc  = AUC_of(data, 'before', 'bad', 1, show=show) if model.groups==FOR.EarlyDischarge or model.groups==FOR.EarlyDischargeO2 else AUC_of(data, show=show)
    pst = show_(S.stats, show=show)
    S(auc=auc, pst=pst)
    if 'bio' in S.stats:
        auc = AUC_of(data, 'bio', show=show)
        S(bio=auc)
    if show: print()
    print('{} virtual patients from {} real subjects assessed'.format(S.virtual, len(S.real)))
    return S

def train_tested(model): model.train(on=1., err=model.err)

def predict(db, from_model, idlabel='hn'):
    DB, mdb, states = db._db, from_model._db, from_model._db.states
    prediction_of = {}
    for idset in idsets_of(DB):
        mdb.states, ID = idset, idset[0].get(idlabel)
        (x,_), ym = from_model.select(**from_model._selargs), []
        for ((M,S),_) in from_model.best:
            try:
                X = array(x)
                Xs = S.transform(X)
                Yp = M.predict(Xs)
                ym.append(Yp)
            except:
                pass
        if len(ym) == 0: ym = -ones((len(from_model.best), len(idset)))
        prediction_of[ID] = array(ym).T
    mdb.states = states
    return prediction_of