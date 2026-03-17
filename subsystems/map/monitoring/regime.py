
import numpy as np

def resolve_regime(s_pos, s_neg, persistence, h):

    n = len(s_pos)

    acc = np.zeros(n, dtype=bool)
    dec = np.zeros(n, dtype=bool)
    osc = np.zeros(n, dtype=bool)

    for t in range(n):

        # oscillatory = never a regime
        if persistence[t] < 0.35:
            osc[t] = True
            continue

        # choose dominant cumulative energy
        if s_pos[t] > s_neg[t]:
            if s_pos[t] > h:
                acc[t] = True
        else:
            if s_neg[t] > h:
                dec[t] = True

    return acc, dec, osc
