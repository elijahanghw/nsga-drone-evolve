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
                hover_status, input_cost, alpha, ctrl, ctrl_eig, volume = zip(*results)

            num_props = [len(drone.props) for drone in drones]

        else:
            hover_status = []
            num_props = []
            input_cost = []
            volume = []
            alpha = []
            ctrl = []
            ctrl_eig = []
            for idx, genotype in enumerate(population):
                drone = get_drone(genotype)
                num_props.append(len(drone.props))

                _hover_status, _input_cost, _alpha, _ctrl, _ctrl_eig, _volume = objective_functions(drone)
                hover_status.append(_hover_status)
                input_cost.append(_input_cost)
                volume.append(_volume)
                alpha.append(_alpha)
                ctrl.append(_ctrl)
                ctrl_eig.append(_ctrl_eig)

        objective_values = list(zip(hover_status, input_cost, alpha, ctrl, ctrl_eig, volume))

        # Save pareto optimal results
        static_drone, static_alpha, static_vol, static_eig, R, F = non_dominated_sort(population, objective_values)

        distance = crowding_distance(static_drone, static_alpha, static_vol, static_eig, F[0])

        print(distance)

        # if file_path is not None:
        #     pop_path = os.path.join(file_path, "population", f"population{i}.npy") 
        #     np.save(pop_path, np.asarray(population))

        #     drone_path = os.path.join(file_path, "drone", f"drone{i}.npy") 
        #     np.save(drone_path, np.asarray(static_drone))

        #     objectives = pd.DataFrame({"alpha": static_alpha, "volume": static_vol, "ctrl": static_eig, "dominated": dominated})
        #     objective_path = os.path.join(file_path, "objective", f"objective{i}.csv") 
        #     objectives.to_csv(objective_path, index=False)

        # while len(new_pop) < pop_size:
        #     ranked_population, ranked_fitness = moga_fitness(population, objective_values)
        #     parent1, parent2 = roulette(ranked_population, ranked_fitness)
        #     child1, child2 = crossover(parent1, parent2)
        #     child1 = mutate(child1)
        #     new_pop.append(child1)

        #     if len(new_pop) == pop_size:
        #         break
        #     else:
        #         child2 = mutate(child2)
        #         new_pop.append(child2)

        # population = new_pop

        # if verbose:
            # print(f"Generation:{i+1}, Best alpha:{max(pareto_alpha):.2f}, Smallest volume:{min(pareto_vol):.4f}, Most ctrl:{max(pareto_eig):.2f}, Num static:{len(static_drone)}, Num pareto:{len(pareto_drone)} Current elapsed time:{(time()-start_time):.2f} \n")

    # print(f"Finished. Total elapsed time: {time()-start_time}")

def moga_fitness(population, objective_values, verbose=0, parallel=True):
    hover_status, input_cost, alpha, ctrl, ctrl_eig, volume = zip(*objective_values)
    fitness = []

    weights = np.random.uniform(low=0, high=1, size=3)
    weights = weights/np.sum(weights)

    for i, _ in enumerate(population):
        if hover_status[i] == "ST":
            hover_score = 10
            alpha_score = alpha[i]
            # ctrl_score = ctrl[i]
            eig_score = ctrl_eig[i]
        elif hover_status[i] == "SP":
            hover_score = 0
            alpha_score = alpha[i]
            # ctrl_score = 0
            eig_score = 0
        elif hover_status[i] == "N":
            hover_score = -10
            alpha_score = 0
            # ctrl_score = 0
            eig_score = 0

        fitness.append(hover_score + weights[0]*0.5*alpha_score + weights[1]*min(eig_score,5) - weights[2]*200*volume[i])

    # Order population
    population_ordered = zip(population, fitness)
    population_ordered = sorted(population_ordered, key=lambda x: x[1], reverse=True)
    population, fitness = zip(*population_ordered)

    population = list(population)
    fitness = list(fitness)

    return population, fitness

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
        return [sim.hover_status, sim.input_cost, sim.alpha, sim.rank_m, min(sim.eig_m), vol]

    elif sim.hover_status == "SP":
        return [sim.hover_status, sim.input_cost, sim.alpha, None, None, vol]

    elif sim.hover_status == "N":
        return [sim.hover_status, None, None, None, None, vol]


def get_drone(genotype):
    phenotype = Phenotype(genotype)
    drone = phenotype.drone
    return drone


def non_dominated_sort(population, objective_values):     
    hover_status, input_cost, alpha, ctrl, ctrl_eig, volume = zip(*objective_values)
    static_drone = [population[i] for i in range(len(population)) if hover_status[i]=="ST"]
    static_alpha = [alpha[i] for i in range(len(population)) if hover_status[i]=="ST"]
    static_vol = [volume[i] for i in range(len(population)) if hover_status[i]=="ST"]
    static_eig = [ctrl_eig[i] for i in range(len(population)) if hover_status[i]=="ST"]

    n = [0 for i in range(len(static_drone))]    # Number of solution dominating solution p
    S = [[] for i in range(len(static_drone))]     # Set of solution which solution i dominates
    Rank = [0 for i in range(len(static_drone))]    # Rank of solution


    F = []      # List of fronts
    F_1 = []    # Set of solution in pareto front

    for p in range(len(static_drone)):
        for q in range(len(static_drone)):
            if static_alpha[p] < static_alpha[q] and static_vol[p] > static_vol[q] and static_eig[p] < static_eig[q]: # p dominated by q
                n[p] = n[p] + 1
            else:   # p NOT dominated by q
                S[p] = list(set().union(S[p], [q]))

        if n[p] == 0:
            Rank[p] = 1
            F_1.append(p)

    F.append(F_1)
    i = 0   # Initialize the front counter

    while len(F[i]) != 0:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    Rank[q] = i + 2
                    Q = list(set().union(Q, [q]))
        i = i + 1
        F.append(Q)

    F.pop()

    return static_drone, static_alpha, static_vol, static_eig, Rank, F

def crowding_distance(static_drone, static_alpha, static_vol, static_eig, front):
    # I = [static_drone[i] for i in front]
    m1 = [static_alpha[i] for i in front]
    m2 = [static_vol[i] for i in front]
    m3 = [static_eig[i] for i in front]
    index = list(range(len(front)))

    objectives = [m1, m2, m3]
    distance = [0 for i in range(len(front))]

    for obj in objectives:
        I_sorted = zip(index, obj)
        I_sorted = sorted(I_sorted, key=lambda x:x[1])
        I_sorted, obj_sorted = zip(*I_sorted)

        for i, idx in enumerate(I_sorted):
            if i == 0 or i == len(front)-1:
                distance[idx] = float('inf')
            else:
                distance[idx] = distance[idx] + (obj_sorted[i+1] - obj_sorted[i-1])/(max(obj_sorted) - min(obj_sorted))

    return distance

