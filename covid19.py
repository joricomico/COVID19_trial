# pylint: disable=no-self-argument, no-member

import sys
sys.path.append('..')

from core import REc, struct
from cop import load_data, translation, patient, model, idsets_of

from numpy import array, ones

def make_model(by_vars, data, groups):
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
    results = _calculate_models_common_init()
    while retries:
        result = _calculate_models_common_calc(hub, calc_args, retries, message)
        ir, dc = set([int(s.hn) for s in hub.in_room]), set([int(s.hn) for s in hub.dismissed])
        result(in_room=ir, discharged=dc)
        retries = _calculate_models_common_redo(hub, result, results, retries)
    return hub, results

def calculate_models_from_start(hub, calc_args, message, retries):
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
    return compute(model, calcargs, message, retry)

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