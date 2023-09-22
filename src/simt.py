import numpy as np
import pandas as pd
from cr import get_connection
from itertools import product

# 0 change
# 2 pdiff (price diff)
# 3 vrate
global acase
acase = 0


def rolling_np(npa, window=5, slide=3):
    i = 0
    length = npa.shape[0]
    res = [] 

    while i + window < length:

        e = npa[i: i+window]
        res.append(e)
        i += slide
        
    return np.array(res)


def rolling_npa(npa, eidx, tkidx, window=5, slide=3):
    i = 0
    npa = npa[eidx, tkidx, :]
    length = npa.shape[0]
    res = [] 

    while i + window < length:

        e = npa[i: i+window]
        res.append(e)
        i += slide
        
    return np.array(res)
    
def model_2(c, p, v, ksp, crc, maths, rxths, rkspths, losscut, w, o, vupper, vlower, sp=0.9976):
    global acase
    acase += 1
    if np.isnan(c).any(): 
        return None

    #print(f'run model: maths:{maths}, rxths:{rxths}, rkspths:{rkspths}, losscut:{losscut}, w:{w}, o:{o}, sp:sp')

    h = 20

    p20 = np.prod(c[:w+o])
    v20 = np.mean(v[:w+o])

    c_x = c[w:w+o]
    p_x = p[w:w+o]
    v_x = v[w:w+o]

    ksp_x = ksp[w:w+o]
    crc_x = crc[w:w+o]

    r_crc_x = np.prod(crc_x)
    r_ksp_x = np.prod(ksp_x)

    rx = np.prod(c_x)
    ry_s = c[w+o:w+o+h]
    assert ry_s.shape[0] == h
    
    ry = np.prod(ry_s)
    ry = max(ry, losscut) 
    ry *= sp

    if p20 > maths and rx > rxths and r_ksp_x > rkspths:
        return ( ry, rx, p20, v20, r_ksp_x)

    return None

def model_1(c, p, v, ksp, crc, mvthres=1.1, rxthres=1.15, rkthres=0.99, losscut=0.91):

    global acase
    acase += 1

    if np.isnan(c).any(): 
        return None

    w = 40

    p20 = np.prod(c[:w])
    v20 = np.mean(v[:w])

    c_x = c[w:w+5]
    p_x = p[w:w+5]
    v_x = v[w:w+5]

    ksp_x = ksp[w:w+5]
    crc_x = crc[w:w+5]

    r_ksp_x = np.prod(ksp_x)
    rx = np.prod(c[w:w+5])
    ry = np.prod(c[w+5:])

    return ry 

    if p20 > mvthres and rx > rxthres:
        #ry = max(ry, losscut)
        return ry * 0.997

    return None

def get_result(series):
    yf = lambda x: x
    win = 0
    loss = 0
    acc = 1 

    for s in series:
        y = yf(s)

        if y > 1:
            win += 1
        else :
            loss += 1
        
    acc = np.prod(series)

    return win, loss, acc


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

def get_param_set(d):
    arr = []

    for k in d :
        init, diff, size = d[k]
        _a = list( init + i*diff for i in range(size) )
        arr.append(_a)

    return [*product(*arr)]

def _s(ts, crc, ksp):   
    con = get_connection()
    _diff = {
        "maths": (1.01, 0.0015, 20),
        "rxths": (1.01, 0.0015, 20),
        "losscut": (0.90, 0.0015, 10),
    }

    sets = get_param_set(_diff)
    w = 60
    sd = 30

    rkspths = 1.016
    w,o = 18 , 9
    vupper = 70
    vlower = 2

    bet_rate = 0.1

    def _get_series(tkidx, **args):
        res = []
        
        c_x = rolling_npa(ts, 0, tkidx,  window=60, slide=sd)
        p_x = rolling_npa(ts, 2, tkidx,  window=60, slide=sd)
        v_x = rolling_npa(ts, 3, tkidx,  window=60, slide=sd)

        ksp_x = rolling_np(ksp, window=60, slide=sd)
        crc_x = rolling_np(crc, window=60, slide=sd)

        for i in range(c_x.shape[0]):
            c = model_2(
                    c_x[i],
                    p_x[i],
                    v_x[i],
                    ksp_x[i],
                    crc_x[i], 
                    **args
            )
            c and res.append(c)

        return res

    for param in sets: 
        maths, rxths, losscut = param

        tres = []
        for k in range(ts.shape[1]):
            r = _get_series(k, maths=maths, rxths=rxths, rkspths=rkspths, losscut=losscut, w=w, o=o, vupper=vupper, vlower=vlower)
            tres.extend(r)

        rys = list(x[0] for x in tres)
        p = bet_md_1(rys, bet_rate)
        
        d = (maths, rxths, sd, bet_rate, losscut, rkspths, w, o, *p)

        df = pd.DataFrame([d])
        df.columns = ('maths', 'rxths', 'sd', 'bet_rate', 'losscut', 'rkspths', 'w', 'o', 'profit', 'casec', 'win', 'loss')
        print(df)
        df.to_sql('mst_2', if_exists='append', con=con)
        

if __name__ == "__main__":
    ts = np.load('data/d__mc5.10.npy')

    crc = np.load('data/cch_mc5.10.npy', allow_pickle=True)
    ksp = np.load('data/ksp_mc5.10.npy', allow_pickle=True)

    ksp = (ksp / 100) + 1
    _s(ts, crc, ksp)
