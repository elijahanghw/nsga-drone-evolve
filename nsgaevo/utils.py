import numba
import numpy as np

def linmap(range, xi):
    return range[0]*(1-xi)/2 + range[1]*(1+xi)/2

def quantmap(vals, xi):
    length = len(vals)
    quantum = 2/length
    for i in range(length):
        if xi <= -1 + (i+1)*quantum:
            return vals[i]

def vector_rotation_matrix(phi,theta):
    x = [np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)]
    x = x/np.linalg.norm(x)
    
    z = [np.sin(-phi)*np.cos(theta), np.sin(-phi)*np.sin(theta), np.cos(phi)]
    z = z/np.linalg.norm(z)
    
    y = np.cross(z, x)
    y = y/np.linalg.norm(y)
    
    R = np.column_stack([x, y, z])
    
    return R