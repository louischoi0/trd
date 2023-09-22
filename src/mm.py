import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import pyplot
import sys 
from sys import exit

wknl = np.log2([1,2,3,4,5])
wknl = np.log2([2,2,2,2,2])

d = {
    'case': 0
}

def _model_1_srt(i, seq, ts, crc, ksp, window_size=5):
    c_x = ts[0, i, seq:seq+window_size]
    p_x = ts[2, i, seq:seq+window_size]
    v_x = ts[3, i, seq:seq+window_size]

    c_y = ts[0, i, seq+window_size:seq+window_size*2]

    mavg10 = ts[0, i, seq-window_size: seq+window_size]
    mavg10 = (np.prod(mavg10) - 1) * 100

    if mavg10 == 0 or mavg10 is None:
        print('mavg is invalid', seq-window_size, seq+window_size)
        print(ts[0, i, seq-window_size: seq+window_size])

    if np.isnan(c_x).any() or np.isnan(c_y).any():
        return None

    ksp_x = (ksp[seq:seq+window_size] / 100) + 1

    if len(c_y) != window_size: return None
    
    d['case'] += 1

    y = np.prod(c_y)

    v_idx0 = np.log10(v_x) * wknl
    v_idx0 = np.sum(v_idx0) * np.prod(ksp_x) 
    
    return v_idx0,  abs(y - 1) * 100 

def model(cidx, ts, crc, ksp):
    res = []
    ref = 100

    for seq in range(10, len(ts[0]), 5):
        x = _model_1_srt(cidx, seq, ts, crc, ksp)
        x and res.append( x )
        #x and print(x)        

    return res
    
if __name__ == "__main__":
    ts = np.load('p.npy')

    crc = np.load('cch.npy', allow_pickle=True)
    ksp = np.load('ksp.npy', allow_pickle=True)
    
    cases = []
    
    for cidx in range(ts.shape[1]):
        r = model(cidx, ts, crc, ksp)

        r = np.array(r)
        s = np.corrcoef(r.T)

        if not np.isnan(s).any(): 
            print(s[0][1])
    
    #pyplot.plot(r)
    #pyplot.show()
