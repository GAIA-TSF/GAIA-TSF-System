
import numpy as np

def resolve_regime(S_pos, S_neg, persistence, h):

    n = len(S_pos)

    acc = np.zeros(n, dtype=bool)
    dec = np.zeros(n, dtype=bool)
    osc = np.zeros(n, dtype=bool)

    for t in range(n):

        # oscillatory = never a regime
        if persistence[t] < 0.35:
            osc[t] = True
            continue

        # choose dominant cumulative energy
        if S_pos[t] > S_neg[t]:
            if S_pos[t] > h:
                acc[t] = True
        else:
            if S_neg[t] > h:
                dec[t] = True

    return acc, dec, osc
