import numpy as np
import pandas as pd
from cr import get_connection
from sys import exit
from itertools import product
import json
from matplotlib import pyplot

def _model_2(state, ts, losscut, w, o, p2th, p1th, atr2th, v2th, k, atrt, sp, **args):
    
    if len(state["positions"]) > 0 : 
        return None
    
    ps, cs, atrs, vs = ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3]

    p2 = np.prod(cs[:w+o])
    v2 = np.mean(vs[:w+o])

    #atr2 = np.mean(atrs[w+o-7:w+o])

    p1 = np.prod(cs[w:w+o])
    v1 = np.mean(vs[w:w+o])

    #atr1 = np.mean(atrs[w+o-3:w+o])

    pa = ps[w+o]

    if p2 > p2th and p1 > p1th and v1 < v2:
        ry = np.prod(cs[w+o:])

        if ry > 1.018:
            print(f'vrt:{v1/v2}, profit: {ry}')

        return pa

    return None

def model_2(state, ts, losscut, w, o, p2th, p1th, atr2th, v2th, k, atrt, sp, **args):
    
    if len(state["positions"]) >= args["maxposition"]: 
        return None
    
    ps, cs, atrs, vs = ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3]

    p2 = np.prod(cs[:w+o])
    v2 = np.mean(vs[:w+o])

    atr2 = np.mean(atrs[w+o-7:w+o])

    p1 = np.prod(cs[w:w+o])
    v1 = np.mean(vs[w:w+o])

    atr1 = np.mean(atrs[w+o-3:w+o])

    pa = ps[w+o]

    if atr1 > 0.2 and atr1 > atr2 * atrt and p2 > p2th and p1 > p1th:
        return pa

    return None

def _gbrt(seed, irt, positions, state, sb):

    weight = (seed / 1000) + 1
    weight = min(1.6, weight)
    weight = 1

    valued = state["valued"]
    
    iseed = min(valued * irt * weight, seed)
    sb['itrm'] += min( irt * weight, iseed / valued )

    #if iseed > 0: print(f'iseed: {iseed} weight: {weight}')

    return iseed

def gbrt(seed, irt, positions, state, sb):
    valued = state["valued"]

    if sb["_sloss"] > 2:
        irt -= (sb["_sloss"] * 5) / 100

    if sb["_swin"] > 2: 
        irt += (sb["_sloss"] * 5) / 100

    iseed = min(valued * irt, seed)
    sb['itrm'] += min( irt, iseed / valued )

    return iseed

def gssgn(state, ts, nidx, positions, mdargs, sb):
    npositions = []

    def sell(s, profit):
        state['seed'] += s * profit
        sb['case'] += 1

    for i, p, s, v in positions:
    
        profit = np.prod( ts[i:nidx, 1] )         
        pr = ( profit - 1 ) * 100 

        if nidx - i >= mdargs['h']:
            sell(s, profit)

            if profit > 1:
                sb["win"] += 1
                sb["wrate"] += pr
                
                sb["_swin"] += 1
                sb["sloss"] = max(sb["sloss"], sb["_sloss"])
                sb["_sloss"] = 0

            else:
                sb["loss"] += 1
                sb["lrate"] += pr
                
                sb["_sloss"] += 1
                sb["swin"] = max(sb["swin"], sb["_swin"])
                sb["_swin"] = 0

        elif profit < mdargs["losscut"]:
            sell(s, profit)

            sb["lrate"] += pr
            sb["losscut"] += 1
            sb["loss"] += 1
            sb["_sloss"] += 1
            sb["swin"] = max(sb["swin"], sb["_swin"])
            sb["_swin"] = 0

        else:
            npositions.append( ( i, p, s, v) )
    return npositions
        
def strts(ts, mdargs):

    state = {
        'seed': 1000,
        'valued': 1000,
        'positions': [],
        'sloss': 0,
        'swin': 0

    } 

    sb = {
        "win": 0,
        "loss": 0,
        "case": 0,
        "_max": 1000,
        "_min": 1000,
        "losscut": 0,
        "wrate": 0,
        "lrate": 0,
        "mdd": 0,

        "_sloss": 0, "_swin": 0,
        "sloss": 0, "swin": 0,
        "_slossa": 0, "_swina": 0,
        "slossa": 0, "swina": 0,


        'itrm': 0,
    }

    w, o, h = mdargs['w'], mdargs['o'], mdargs['h']
    ws =  mdargs['w'] + mdargs['o'] + mdargs['h']

    mts = []
    
    for i in range(w+o, len(ts) - h):
        _ts = ts[i-w-o: i+h]
        signal_price = model_2(state, _ts, **mdargs)

        p = ts[i][0]
        chg = ts[i][1]

        if state["positions"]: 
            state['positions'] = gssgn(state, ts, i, state['positions'], mdargs, sb)

        if signal_price and state['seed'] > 10:
            iseed = gbrt(state['seed'], mdargs['irt'], state['positions'], state, sb)

            state['seed'] -= iseed
            state['positions'].append( (i, p, iseed, iseed / p ) )
            #print(f'got signal!, sp:{signal_price} is:{iseed}')
        
        vp = sum( x[3] * p for x in state["positions"] ) + state["seed"]
        state["valued"] = vp

        sb["_max"] = max(state["valued"], sb["_max"])
        sb["_min"] = min(state["valued"], sb["_min"])
        sb["mdd"] = max( ((sb["_max"] - state["valued"]) / sb["_max"] ), sb["mdd"] )
        mts.append(state["valued"])
    
    sb["wrate"] /= sb["win"]
    sb["lrate"] /= sb["loss"]
    sb["itrm"] /= sb["case"]
    sb["plrate"] = sb["wrate"] / abs(sb["lrate"])
    
    return state, sb, mts

def get_param_set(defaults, d):
    arr = []
    sets = []

    for k in d :
        init, diff, size = d[k]
        _a = list( (k, init + i*diff) for i in range(size) )

        arr.append(_a)

    for s in product(*arr):
        _d = defaults.copy()
        for k,v in s:
            _d[k] = v

        sets.append(_d)

    return sets

def get_paramsets(redis, stratname):

    while True:
        print('get')
        v = redis.lpop(stratname)

        if v:
            v = v.decode('utf8')
            v = json.loads(v)
        yield v

def get_redis():
    import redis
    return redis.Redis(host='localhost', port=6379, db=0)

if __name__ == "__main__":
    con = get_connection()
    ts = np.load('data/btc.npy')
    ts = ts[1:]

    _diff = {
        "atrt": (1.05, 0.025, 5),
        "p2th": (1.03, 0.0015, 20),
        "p1th": (1.005, 0.00025, 10),
    }
    _diff = {}

    mdargs = {
        'sp': 0.977,
        'irt': 0.48,
        'atrt': 1.08,
        'p2th': 1.02,
        'p1th': 1.006,
        'losscut': 0.975,
        'atr2th': 0.1,
        'v2th': 0.1,
        'k': 0.5,
        'w': 60 * 7,
        'o': 60 * 3,
        'h': 60 * 8,
        'maxposition': 3
    }

    sets = get_param_set(mdargs, _diff)
    redis = get_redis()

    #for _ms in get_paramsets(redis, 'mmv1'):
    for _ms in sets:
        print('ms:', _ms)
        if _ms is None: break
        #del _ms["_hash"]
        
        state, sb, mts = strts(ts, _ms)
        d = { **state, **sb, **_ms }

        d['profit'] = d['seed'] / 1000

        del d['valued']
        del d['seed']
        del d['positions']

        df = pd.DataFrame([d])
        df['createdAt'] = pd.Timestamp.now()
        #df.to_sql('btc_mmv1_seq', if_exists='append', con=con, index=False)
        #print(df)
        print({ **sb, 'profit': d['profit'] })
        pyplot.plot(mts)
        pyplot.show()
        break

