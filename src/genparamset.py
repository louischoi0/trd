import collections
from itertools import product

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

def hashparams(p):
    od = collections.OrderedDict(sorted(p.items()))
    print(od)
    s = ""
    for k in od:
        s += k
        s += str(od[k])
    
    return hash(s)

if __name__ == "__main__":
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

    h1 = hashparams(sets[0])
    h2 = hashparams(sets[0])
    
    print(h1)
    print(h2)
