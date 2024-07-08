import os
from time import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from multiprocessing import Pool
from signal import signal, SIGINT
from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.algorithms.ga_utils import *

def initializer():
    signal(SIGINT, lambda: None)

def nsga(population, num_gen, verbose=True, eval_verbose=0, file_path=None, parallel=True):
    print("Starting NSGA-II...\n")
    start_time = time()
    if file_path is not None:
        if not os.path.exists(os.path.join(file_path, "population")):
            os.mkdir(os.path.join(file_path, "population"))
        if not os.path.exists(os.path.join(file_path, "drone")):
            os.mkdir(os.path.join(file_path, "drone"))
        if not os.path.exists(os.path.join(file_path, "objective")):
            os.mkdir(os.path.join(file_path, "objective"))

    pop_size = len(population)

    for i in range(num_gen):

        new_pop = []

        if parallel:
            with Pool(20, initializer=initializer) as pool: 
                drones = pool.map(get_drone, population)
                results = pool.map(objective_functions, drones)
                hover_status, alpha, ctrl_eig, volume = zip(*results)

            num_props = [len(drone.props) for drone in drones]

        else:
            hover_status = []
            num_props = []
            volume = []
            alpha = []
            ctrl_eig = []
            for idx, genotype in enumerate(population):
                drone = get_drone(genotype)
                num_props.append(len(drone.props))

                _hover_status, _alpha, _ctrl_eig, _volume = objective_functions(drone)
                hover_status.append(_hover_status)
                volume.append(_volume)
                alpha.append(_alpha)
                ctrl_eig.append(_ctrl_eig)

        objective_values = list(zip(hover_status, alpha, ctrl_eig, volume))

        # Save pareto optimal results
        R, F = non_dominated_sort(population, objective_values)

        distance = crowding_distance(objective_values, F)

        print(distance)



def objective_functions(drone):
    # Compute size of bounding box using PCA

    points = []
    for prop in drone.props:
        points.append(prop["loc"])
    points = np.array(points)

    points_gram = points.T @ points

    w, v = np.linalg.eig(points_gram)

    points_transformed = points @ v
    
    min_coords = np.min(points_transformed, axis=0)
    max_coords = np.max(points_transformed, axis=0)
    dim = max_coords - min_coords
    if dim[0] <= 0.06:     # Add 6cm thickness to "planar" builds
        dim[0] = 0.06
    if dim[1] <= 0.06:     # Add 6cm thickness to "planar" builds
        dim[1] = 0.06
    if dim[2] <= 0.06:     # Add 6cm thickness to "planar" builds
        dim[2] = 0.06

    vol = np.prod(dim)
    # vol = max(dim)

    sim = Hover(drone)
    sim.compute_hover()

    if sim.hover_status == "ST":
        return [sim.hover_status, sim.alpha, min(sim.eig_m), vol]

    elif sim.hover_status == "SP":
        return [sim.hover_status, sim.alpha, 0, vol]

    elif sim.hover_status == "N":
        return [sim.hover_status, 0, 0, vol]


def get_drone(genotype):
    phenotype = Phenotype(genotype)
    drone = phenotype.drone
    return drone


def non_dominated_sort(population, objective_values):     
    hover_status, alpha, ctrl_eig, volume = zip(*objective_values)

    static_drone = [i for i in range(len(population)) if hover_status[i]=="ST"]
    spinning_drone = [i for i in range(len(population)) if hover_status[i]=="SP"]
    nohover_drone = [i for i in range(len(population)) if hover_status[i]=="N"]

    n = [None for i in range(len(population))]    # Number of solution dominating solution p
    S = [None for i in range(len(population))]     # Set of solution which solution i dominates
    rank = [0 for i in range(len(population))]    # Rank of solution

    F = []      # List of fronts
    F_1 = []    # Set of solution in pareto front

    # Nondominated sort for static drones
    for p in static_drone:
        n[p] = 0
        S[p] = []
        for q in static_drone:
            if alpha[p] < alpha[q] and volume[p] > volume[q] and ctrl_eig[p] < ctrl_eig[q]: # p dominated by q
                n[p] = n[p] + 1
            else:   # p NOT dominated by q
                S[p] = list(set().union(S[p], [q]))

        if n[p] == 0:
            rank[p] = 1
            F_1.append(p)

    F.append(F_1)

    i = 0   # Initialize the front counter

    while len(F[i]) != 0:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 2
                    Q = list(set().union(Q, [q]))
        i = i + 1
        F.append(Q)
    F.pop()

    Q_1 = []
    # Nondominated sort for spinning drones
    for p in spinning_drone:
        n[p] = 0
        S[p] = []
        for q in spinning_drone:
            if alpha[p] < alpha[q] and volume[p] > volume[q]: # p dominated by q
                n[p] = n[p] + 1
            else:   # p NOT dominated by q
                S[p] = list(set().union(S[p], [q]))

        if n[p] == 0:
            rank[p] = i
            Q_1.append(p)

    F.append(Q_1)

    while len(F[i]) != 0:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 2
                    Q = list(set().union(Q, [q]))
        i = i + 1
        F.append(Q)
    F.pop()

    F.append(nohover_drone)

    max_rank = max(rank)

    for i in nohover_drone:
        rank[i] = max_rank + 1

    return rank, F

def crowding_distance(objective_values, fronts):
    hover_status, alpha, ctrl_eig, volume = zip(*objective_values)
    distance = [0 for i in range(len(hover_status))]
    for f in fronts:
        m1 = [alpha[i] for i in f]
        m2 = [volume[i] for i in f]
        m3 = [ctrl_eig[i] for i in f]

        objectives = [m1, m2, m3]

        for obj in objectives:
            if (max(obj) - min(obj)) == 0:
                pass
            else:
                I_sorted = zip(f, obj)
                I_sorted = sorted(I_sorted, key=lambda x:x[1])
                I_sorted, obj_sorted = zip(*I_sorted)

                for i, idx in enumerate(I_sorted):
                    if i == 0 or i == len(f)-1:
                        distance[idx] = float('inf')
                    else:
                        distance[idx] = distance[idx] + (obj_sorted[i+1] - obj_sorted[i-1])/(max(obj_sorted) - min(obj_sorted))

    return distance

