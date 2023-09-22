import numpy as np
import pandas as pd
from cr import get_connection
from sys import exit
from btcmd import model_2
from itertools import product

def model_2(state, ts, losscut, w, o, p2th, p1th, atr2th, v2th, k, atrt, sp, **args):
    
    if len(state["positions"]) > 2 : 
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

def gbrt(seed, irt, positions, state):
    
    weight = (seed / 1000) + 1
    weight = min(1.6, weight)
    weight = 1

    valued = state["valued"]
    
    iseed = min(valued * irt * weight, seed)

    #if iseed > 0: print(f'iseed: {iseed} weight: {weight}')

    return iseed

def gssgn(state, ts, nidx, positions, mdargs, sb):
    npositions = []

    def sell(s, profit):
        state['seed'] += s * profit
        sb['case'] += 1

    for i, p, s, v in positions:
    
        profit = np.prod( ts[i:nidx, 1] )         

        if nidx - i >= mdargs['h']:
            sell(s, profit)

            if profit > 1:
                sb["win"] += 1
            else:
                sb["loss"] += 1

        elif profit < mdargs["losscut"]:
            sb["losscut"] += 1
            sb["loss"] += 1
            sell(s, profit)

        else:
            npositions.append( ( i, p, s, v) )
    
    return npositions
        
def strts(ts, mdargs):

    state = {
        'seed': 1000,
        'valued': 1000,
        'positions': []
    } 

    sb = {
        "win": 0,
        "loss": 0,
        "case": 0,
        "_max": 1000,
        "_min": 1000,
        "losscut": 0,
        "mdd": 0
    }

    w, o, h = mdargs['w'], mdargs['o'], mdargs['h']
    ws =  mdargs['w'] + mdargs['o'] + mdargs['h']
    
    for i in range(w+o, len(ts) - h):
        _ts = ts[i-w-o: i+h]
        signal_price = model_2(state, _ts, **mdargs)

        p = ts[w+o][0]
        state["valued"] = sum( x[3] * p for x in state["positions"] ) + state["seed"]
    
        if signal_price:
            iseed = gbrt(state['seed'], mdargs['irt'], state['positions'], state)

            if iseed < 10: continue

            state['seed'] -= iseed
            state['positions'].append( (i, p, iseed, iseed / signal_price ) )
            #print(f'got signal!, sp:{signal_price} is:{iseed}')

        if state["positions"]: 
            state['positions'] = gssgn(state, ts, i, state['positions'], mdargs, sb)

        sb["_max"] = max(state["valued"], sb["_max"])
        sb["_min"] = min(state["valued"], sb["_min"])
        sb["mdd"] = max( ((sb["_max"] - state["valued"]) / sb["_max"] ), sb["mdd"] )

    return state, sb

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

if __name__ == "__main__":
    con = get_connection()
    ts = np.load('data/btc.npy')
    ts = ts[1:]

    _diff = {
        "atrt": (1.05, 0.025, 5),
        "p2th": (1.03, 0.0015, 20),
        "p1th": (1.005, 0.00025, 10),
    }

    mdargs = {
        'sp': 0.977,
        'irt': 0.48,
        'atrt': 1.08,
        'p2th': 1.028,
        'p1th': 1.007,
        'losscut': 0.975,
        'atr2th': 0.1,
        'v2th': 0.1,
        'k': 0.5,
        'w': 60 * 11,
        'o': 60 * 3,
        'h': 60 * 9
    }

    sets = get_param_set(mdargs, _diff)

    for _ms in sets:
        state, sb = strts(ts, _ms)
        d = { **state, **sb, **_ms }

        d['profit'] = d['seed'] / 1000

        del d['valued']
        del d['seed']
        del d['positions']

        df = pd.DataFrame([d])
        df['createdAt'] = pd.Timestamp.now()
        df.to_sql('btc_mst_md2_seq', if_exists='append', con=con, index=False)
        print(df)

        

    





