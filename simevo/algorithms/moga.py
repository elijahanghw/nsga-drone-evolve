import os
from time import time
import numpy as np
from multiprocessing import Pool
from signal import signal, SIGINT
from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.algorithms.ga_utils import *

def initializer():
    signal(SIGINT, lambda: None)

def moga(population, num_gen, verbose=True, eval_verbose=0, file_path=None, parallel=True):
    print("Starting genetic algorithm...\n")
    start_time = time()
    if file_path is not None:
        if not os.path.exists(os.path.join(file_path, "population")):
            os.mkdir(os.path.join(file_path, "population"))
        if not os.path.exists(os.path.join(file_path, "fitness")):
            os.mkdir(os.path.join(file_path, "fitness"))
    pop_size = len(population)
    for i in range(num_gen):
        new_pop = []

        if parallel:
            with Pool(20, initializer=initializer) as pool: 
                drones = pool.map(geno_to_pheno, population)
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
                drone = Phenotype(genotype)
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

        while len(new_pop) < pop_size:
            ranked_population, ranked_fitness = moga_fitness(population, objective_values)
            parent1, parent2 = roulette(ranked_population, ranked_fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            new_pop.append(child1)

            if len(new_pop) == pop_size:
                break
            else:
                child2 = mutate(child2)
                new_pop.append(child2)

        population = new_pop

        if verbose:
            print(f"Generation:{i+1}, Max fitness:{ranked_fitness[0]:.2f}, Mean fitness:{np.mean(ranked_fitness):.2f},  Current elapsed time:{time()-start_time} \n")

    print(f"Finished. Total elapsed time: {time()-start_time}")

def moga_fitness(population, objective_values, verbose=0, parallel=True):
    hover_status, _, alpha, _, _, volume = zip(*objective_values)
    fitness = []

    weights = np.random.uniform(low=0, high=1, size=3)
    weights = weights/np.linalg.norm(weights)

    for i, _ in enumerate(population):
        if hover_status[i] == "ST":
            hover_score = 10
            alpha_score = alpha[i]
        elif hover_status[i] == "SP":
            hover_score = 0
            alpha_score = 0
        elif hover_status[i] == "N":
            hover_score = -10
            alpha_score = 0

        fitness.append(weights[0]*hover_score + weights[1]*alpha_score + weights[2]*volume[i])

    # Order population
    population_ordered = zip(population, fitness)
    population_ordered = sorted(population_ordered, key=lambda x: x[1], reverse=True)
    population, fitness = zip(*population_ordered)

    population = list(population)
    fitness = list(fitness)

    return population, fitness

def objective_functions(drone):
    # Compute size of bounding box: Now just x-y-z, to use PCA in future
    points = []
    for prop in drone.props:
        points.append(prop["loc"])

    points = np.array(points)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dim = max_coords - min_coords
    if dim[0] == 0:     # Add 5cm thickness to "planar" builds
        dim[0] = 0.05
    if dim[1] == 0:     # Add 5cm thickness to "planar" builds
        dim[1] = 0.05
    if dim[2] == 0:     # Add 5cm thickness to "planar" builds
        dim[2] = 0.05

    vol = np.prod(dim)

    sim = Hover(drone)
    sim.compute_hover()

    weights = np.random.uniform(low=0, high=1, size=2)
    weights = weights/np.linalg.norm(weights)
    if sim.hover_status == "ST":
        # fit = 10 - 100*sim.input_cost + weights[0]*0.5*sim.alpha - weights[1]*1000*vol + sim.rank_m + min(np.prod(sim.eig_m/1000000),3)
        return [sim.hover_status, sim.input_cost, sim.alpha, sim.rank_m, min(np.prod(sim.eig_m/1000000),3), vol]

    elif sim.hover_status == "SP":
        # fit = 0 - 100*sim.input_cost + weights[0]*0.5*sim.alpha - weights[1]*1000*vol
        return [sim.hover_status, sim.input_cost, sim.alpha, None, None, vol]

    elif sim.hover_status == "N":
        # fit = -10 - 1000*vol
        return [sim.hover_status, None, None, None, None, vol]


def geno_to_pheno(genotype):
    drone = Phenotype(genotype)
    return drone

def pareto_optimal(population, objective_values):
    hover_status, _, alpha, _, _, volume = zip(*objective_values)

    static_drone = [drone for drone in population if "ST" in hover_status]
    static_alpha = [a for a in alpha if "ST" in hover_status]
    static_vol = [vol for vol in volume if "ST" in hover_status]

    optimal_drone = []

    for (drone, a, vol) in zip(static_drone, static_alpha, static_vol):
        continue

    return optimal_drone