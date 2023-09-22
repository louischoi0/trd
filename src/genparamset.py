import collections
from itertools import product
from cr import get_connection
import json
import redis
import hashlib

def get_redis():
    return redis.Redis(host='localhost', port=6379, db=0)


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

        _d['_hash'] = hashparams(_d)
        yield _d
        #sets.append(_d)

    #return sets

def hashparams(p):
    od = collections.OrderedDict(sorted(p.items()))

    s = ""
    for k in od:
        s += k
        s += str(od[k]) + ":"

    s = s.encode('utf8') 
    return hashlib.md5(s).hexdigest()

if __name__ == "__main__":
    stratname = 'mmv1'
    _diff = {
        "atrt": (1.05, 0.025, 5),
        "p2th": (1.02, 0.0015, 30),
        "p1th": (1.003, 0.00025, 30),
        "h": (7, 1, 6),
        "o": (3, 1, 5),
        "w": (7, 1, 6),
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
    r = get_redis()
    i = 0

    for ss in sets:
        i += 1
        ss = json.dumps(ss)
        r.lpush(stratname, ss)
        print(ss)
        if i > 100: break
