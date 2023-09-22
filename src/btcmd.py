import numpy as np
import pandas as pd
from cr import get_connection
from itertools import product
from sys import exit

# price, change, atr, volume

def bet_md_1(rys, irt):
    casec = 0
    win = 0
    loss = 0

    seed = 1000

    for ry in rys:
        inv = seed * irt
        seed -= inv
        seed += inv * ry
    
        casec += 1
        if ry > 1:
            win += 1
        else:
            loss += 1

    return (seed / 1000, casec, win, loss )

def randomsample_np(npa, window, count):
    length = npa.shape[0]

    res = []

    for i in range(count):
        _w = np.random.randint(0, length - window)
        res.append( npa[_w:_w+window] )

    return np.concatenate(res, axis=0)

def rolling_np(npa, window=5, slide=3):
    i = 0
    length = npa.shape[0]
    res = [] 

    while i + window < length:

        e = npa[i: i+window]
        res.append(e)
        i += slide
        
    return np.array(res)

def sellstrat(ys, atrs, iatr):
    _y =  np.prod(ys[:5])
    w = ( x * 30 for x in range(7*2) )

    for _w in w:
        _atr = np.mean(atrs[:_w])

        if _atr < iatr * 0.35:
            return np.prod(ys[:_w])
        
    return np.prod(ys)

def model_3(ts, losscut, w, o, p2th, p1th, atr2th, v2th, k, atrt, sp):
    ps, cs, atrs, vs = ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3]

    p2 = np.prod(cs[:w+o])
    v2 = np.mean(vs[:w+o])
    atr2 = np.mean(atrs[w+o-7:w+o])

    p1 = np.prod(cs[w:w+o])
    v1 = np.mean(vs[w:w+o])
    atr1 = np.mean(atrs[w+o-3:w+o])

    pa = ps[w+o]

    pt = np.mean(ps[w:w+o])
    pt = pt + ( (pt/25) * atr1 * k)

    if atr1 > 0.1 and atr1 > atr2 * atrt and p2 > p2th and p1 > p1th and pa > pt :
        y_s = cs[w+o:]
        atr_s = atrs[w+o:]
        #ry = sellstrat(y_s, atr_s, atr1)
        ry = np.prod(y_s)
        #ry = max(ry, losscut) 

        ry *= sp
        
        return ry

    return None

def model_2(ts, losscut, w, o, p2th, p1th, atr2th, v2th, k, atrt, sp):
    ps, cs, atrs, vs = ts[:, 0], ts[:, 1], ts[:, 2], ts[:, 3]

    p2 = np.prod(cs[:w+o])
    v2 = np.mean(vs[:w+o])
    atr2 = np.mean(atrs[w+o-7:w+o])

    p1 = np.prod(cs[w:w+o])
    v1 = np.mean(vs[w:w+o])
    atr1 = np.mean(atrs[w+o-3:w+o])

    pa = ps[w+o-1]

    if atr1 > 0.1 and atr1 > atr2 * atrt and p2 > p2th and p1 > p1th:
        y_s = cs[w+o:]
        atr_s = atrs[w+o:]
        #ry = sellstrat(y_s, atr_s, atr1)
        ry = np.prod(y_s)
        #ry = max(ry, losscut) 

        ry *= sp
        
        return ry

    return None

def get_param_set(d):
    arr = []

    for k in d :
        init, diff, size = d[k]
        _a = list( init + i*diff for i in range(size) )
        arr.append(_a)

    return [*product(*arr)]

if __name__ == "__main__":
    con = get_connection()
    ts = np.load('data/btc.npy')
    ts = ts[1:]

    sd = 30
    w, o, h = 60*11, 60*3, 60*11

    ws = w + o + h

    rts = rolling_np(ts, ws, sd)
    #rts = randomsample_np(ts, 50, 100)

    _diff = {
        #"k": (0.5, 0.025, 30),
        #"losscut": (0.975, 0.001, 10),
        "atrt": (1.05, 0.025, 5),
        "p2th": (1.033, 0.0015, 20),
        "p1th": (1.001, 0.00025, 10),
        #"irt": (0.47, 0.01, 5),
    }

    sets = get_param_set(_diff)
    sp = 0.9977

    for s in sets:
        atrt, p2th, p1th = s

        irt = 0.51
        k = None
        losscut = None
        v2th = None
        atr2th = None

        rs = []
        tcase = 0
        for t in rts:
            tcase += 1
            r = model_2(
                    t,
                    losscut=losscut,
                    v2th=v2th,
                    p2th=p2th,
                    p1th=p1th,
                    atr2th=atr2th,
                    atrt=atrt,
                    k=k,
                    sp=sp,
                    w=w,
                    o=o
            )
            r and rs.append(r)
        
        sr = bet_md_1(rs, irt)

        d = (w/60, o/60, h/60,irt, k, losscut, atrt, v2th, p2th, p1th, atr2th, tcase, *sr)
        df = pd.DataFrame([d])

        df.columns = [ 'w', 'o', 'h', 'irt', 'k', 'losscut', 'atrt', 'v2th', 'p2th', 'p1th', 'atr2th','tcase','profit',  'casec', 'win', 'loss']
        df.to_sql('btc_mst_4_md2', if_exists='append', con=con)
        print(df)
     

