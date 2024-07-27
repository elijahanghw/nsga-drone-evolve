import numba

def linmap(range, xi):
    return range[0]*(1-xi)/2 + range[1]*(1+xi)/2

def quantmap(vals, xi):
    length = len(vals)
    quantum = 2/length
    for i in range(length):
        if xi <= -1 + (i+1)*quantum:
            return vals[i]
