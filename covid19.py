# pylint: disable=no-self-argument, no-member

import sys
sys.path.append('..')

from core import REc
from core_legacy import some, no
from cop import load_data, translation, patient, model

def make_model(by_vars, data, groups):
    T = translation(by_vars)
    pclass = patient(by_vars)
    pclass.load(data, T)
    return model(pclass, groups=groups)

def test_calculate_models(hub, calc_args, message, retries):
    results = []
    while retries:
        print('{} tries before saving...'.format(retries))
        print(message.format(hub.DB.ids, hub.DB.long_ids))
        hub.calculate(**calc_args)
        hub.best = hub.unstacked
        assessed = set([int(s.hn) for s in hub.assessed])
        result = struct(vars=hub.vars_of, stats=hub.stats, assessed=assessed)
        hub.best, hub.unstacked = [], []
        results.append(result)
        retries -= 1
    return results    

class FOR:
    EarlyDischarge = {(0,):0, range(1,5):1}
    EarlyDischargeO2 = {(0,1):0, range(2,5):1}
    Worsening = {(0,1):0, range(2,5):1}
    MechVent = {(0,1,2):0, range(3,5):1}
    ICU = {(0,1,2):0, range(3,6):1}

def test(varfile, csvdb, groups=FOR.EarlyDischarge, message='', errstart=.05, retry_model=10):
    model, errstart = make_model(varfile, csvdb, groups)
    calcargs = dict(on=.6, virtual=int(model.DB.long_ids*.3), outcome_from=3, err=err, skip=0)
    results = test_calculate_models(model, calcargs, 'dcr-predict MR: {} subjects, {} in room', retry_model)
