import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from signal import signal, SIGINT
from dronehover.optimization import Hover
from dronehover.bodies.standard_bodies import Quadcopter
from scipy.spatial import ConvexHull

from nsgaevo.phenotype import Phenotype
from nsgaevo.algorithms.ga_utils import *

def initializer():
    signal(SIGINT, lambda: None)

def nsga(population, num_gen, verbose=True, eval_verbose=0, file_path=None, parallel=True):
    if file_path is not None:
        if not os.path.exists(os.path.join(file_path, "population")):
            os.mkdir(os.path.join(file_path, "population"))
        if not os.path.exists(os.path.join(file_path, "objectives")):
            os.mkdir(os.path.join(file_path, "objectives"))
    print("Starting NSGA-II...\n")
    start_time = time()

    pop_size = len(population)

    hover, alpha, ctrl, size, num_props = get_objectives(population, parallel=parallel)    

    objective_values = list(zip(hover, alpha, ctrl, size))

    # Sort by non-dominance and compute distance measure
    rank, fronts = non_dominated_sort(population, objective_values)
    distance = crowding_distance(objective_values, fronts)

    #  Save initial population
    objective_save = pd.DataFrame({"rank": rank, "hover": hover, "alpha": alpha, "size": size, "ctrl": ctrl, "num_props": num_props})
    if file_path is not None:
        pop_path = os.path.join(file_path, "population", f"initial_population.npy") 
        np.save(pop_path, np.asarray(population))
        objective_path = os.path.join(file_path, "objectives", f"initial_objectives.csv")
        objective_save.to_csv(objective_path, index=False)

    for gen in range(num_gen):
        # Populating extended_pop using crossover and mutation
        extended_pop = []
        while len(extended_pop) < pop_size:
            parent1 = tournament(rank, distance)
            parent2 = None
            while parent1 == parent2 or parent2 == None:
                parent2 = tournament(rank,distance)
            child1, child2 = crossover(population[parent1], population[parent2])
            child1 = mutate(child1)
            extended_pop.append(child1)

            if len(extended_pop) == pop_size:
                break
            else:
                child2 = mutate(child2)
                extended_pop.append(child2)

        extended_hover, extended_alpha, extended_ctrl, extended_size, extended_num_props = get_objectives(extended_pop, parallel=parallel)  

        # Combine population and objectives
        combined_pop = population + extended_pop
        combined_hover = hover + extended_hover
        combined_alpha = alpha + extended_alpha
        combined_size = size + extended_size
        combined_ctrl = ctrl + extended_ctrl

        combined_num_props = num_props + extended_num_props

        objective_values = list(zip(combined_hover, combined_alpha, combined_ctrl, combined_size))

        # Sort by non-dominance and compute distance measure
        combined_rank, combined_fronts = non_dominated_sort(combined_pop, objective_values)

        combined_distance = crowding_distance(objective_values, combined_fronts)

        population = []
        rank = []
        distance = []

        hover = []
        alpha = []
        size = []
        ctrl = []

        num_props = []
        
        # Populate new population
        count = 0
        while len(population) + len(combined_fronts[count]) < pop_size:
            for idx in combined_fronts[count]:
                population.append(combined_pop[idx])
                rank.append(combined_rank[idx])
                distance.append(combined_distance[idx])

                hover.append(combined_hover[idx])
                alpha.append(combined_alpha[idx])
                size.append(combined_size[idx])
                ctrl.append(combined_ctrl[idx])

                num_props.append(combined_num_props[idx])

            if count == 0:
                optimal_alpha = max(alpha)
                optimal_size = min(size)
                optimal_ctrl = max(ctrl)

            count = count + 1

        front_distance = [combined_distance[i] for i in combined_fronts[count]]
        sorted_front = zip(combined_fronts[count], front_distance)
        sorted_front = sorted(sorted_front, key=lambda x:x[1], reverse=True)
        sorted_front, front_distance = zip(*sorted_front)
        sorted_front = list(sorted_front)

        for idx in sorted_front:
            if len(population) < pop_size:
                population.append(combined_pop[idx])
                rank.append(combined_rank[idx])
                distance.append(combined_distance[idx])

                hover.append(combined_hover[idx])
                alpha.append(combined_alpha[idx])
                size.append(combined_size[idx])
                ctrl.append(combined_ctrl[idx])

                num_props.append(combined_num_props[idx])

            else:
                break

        if count == 0:
            optimal_alpha = max(alpha)
            optimal_size = min(size)
            optimal_ctrl = max(ctrl)


        # Save new population
        objective_save = pd.DataFrame({"rank": rank, "hover": hover, "alpha": alpha, "size": size, "ctrl": ctrl, "num_props": num_props, "distance": distance})
        if file_path is not None:
            pop_path = os.path.join(file_path, "population", f"population{gen+1}.npy") 
            np.save(pop_path, np.asarray(population))
            objective_path = os.path.join(file_path, "objectives", f"objectives{gen+1}.csv")
            objective_save.to_csv(objective_path, index=False)

        print(f"Generation {gen+1}    Num optimal: {len(combined_fronts[0])}    Max alpha: {optimal_alpha:.2f}    Min vol: {optimal_size:.4f}    Max ctrl:{optimal_ctrl:.2f}   Time elapsed:{time()-start_time:.2f}")


def get_objectives(population, parallel=True):
    if parallel:
        with Pool(initializer=initializer) as pool: 
            drones = pool.map(get_drone, population)
            results = pool.map(objective_functions, drones)
            hover, alpha, ctrl, size = zip(*results)

        num_props = [len(drone.props) for drone in drones]

    else:
        hover = []
        num_props = []
        size = []
        alpha = []
        ctrl = []
        for idx, genotype in enumerate(population):
            drone = get_drone(genotype)
            num_props.append(len(drone.props))

            _hover, _alpha, _ctrl, _size = objective_functions(drone)
            hover.append(_hover)
            size.append(_size)
            alpha.append(_alpha)
            ctrl.append(_ctrl)
    
    return list(hover), list(alpha), list(ctrl), list(size), list(num_props)

def objective_functions(drone):
    # Compute area of convex hull
    points = []
    for prop in drone.props:
        points.append(prop["loc"])
    points.append([0, 0, 0])    # Include origin
    points = np.array(points)
    hull = ConvexHull(points[:,0:2])
    vol = hull.volume

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
    hover, alpha, ctrl, size = zip(*objective_values)

    static_drone = [i for i in range(len(population)) if hover[i]=="ST"]
    spinning_drone = [i for i in range(len(population)) if hover[i]=="SP"]
    nohover_drone = [i for i in range(len(population)) if hover[i]=="N"]

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
            
            if alpha[p] >= alpha[q] and size[p] <= size[q] and ctrl[p] >= ctrl[q] \
                and (alpha[p] > alpha[q] or size[p] < size[q] or ctrl[p] > ctrl[q]):   # if p dominates by q
                S[p] = list(set().union(S[p], [q]))
            
            elif alpha[p] <= alpha[q] and size[p] >= size[q] and ctrl[p] <= ctrl[q] \
                and (alpha[p] < alpha[q] or size[p] > size[q] or ctrl[p] < ctrl[q]): # q dominates by p
                n[p] = n[p] + 1

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
    if len(spinning_drone) > 0:
        for p in spinning_drone:
            n[p] = 0
            S[p] = []
            for q in spinning_drone:
                if alpha[p] >= alpha[q] and size[p] <= size[q] and (alpha[p] > alpha[q] or size[p] < size[q]):   # if p dominates q
                    S[p] = list(set().union(S[p], [q]))
                elif alpha[p] <= alpha[q] and size[p] >= size[q] and (alpha[p] < alpha[q] or size[p] > size[q]): # q dominates by p
                    n[p] = n[p] + 1

            if n[p] == 0:
                rank[p] = i + 1
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

    if len(nohover_drone) > 0:
        F.append(nohover_drone)

    max_rank = max(rank)

    for i in nohover_drone:
        rank[i] = max_rank + 1

    return rank, F

def crowding_distance(objective_values, fronts):
    hover, alpha, ctrl, size = zip(*objective_values)
    distance = [0 for k in range(len(hover))]
    for f in fronts:
        m1 = [alpha[i] for i in f]
        m2 = [size[i] for i in f]
        m3 = [ctrl[i] for i in f]

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

