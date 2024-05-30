import numba

def linmap(range, xi):
    return range[0]*(1-xi)/2 + range[1]*(1+xi)/2