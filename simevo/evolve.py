import os
from time import time
import numpy as np
from dronehover.optimization import Hover

from simevo import min_props, max_props
from simevo.genotype import get_genotype
from simevo.phenotype import Phenotype

def generate_population(pop_size, min_props=min_props, max_props=max_props):
    population = []
    genotype_len = 5*min_props + (5+1)*(max_props-min_props)
    for _ in range(pop_size):
        genotype = get_genotype(genotype_len)
        population.append(genotype)
    return population


def genetic_algorithm(population, num_gen, verbose=True, eval_verbose=0, file_path=None):
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

        ranked_population, ranked_fitness = evaluate_fitness(population, verbose=eval_verbose)

        if file_path is not None:
            pop_path = os.path.join(file_path, "population", f"population{i}.npy")
            fit_path = os.path.join(file_path, "fitness", f"fitness{i}.npy") 
            np.save(pop_path, np.asarray(ranked_population))
            np.save(fit_path, np.asarray(ranked_fitness))

        # Bring over the top individuals
        for j in range(5):
            new_pop.append(ranked_population[j])

        while len(new_pop) < pop_size:
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

    # return population

def evaluate_fitness(population, verbose=0):
    hover_status = []
    num_props = []
    input_cost = []
    volume = []
    alpha = []
    fitness = []
    for idx, genotype in enumerate(population):
        drone = Phenotype(genotype)
        num_props.append(len(drone.props))
        # Compute size of bounding box
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

        if sim.hover_status == "ST":
            fit = 10 - 50*sim.input_cost + sim.alpha + 0.01/vol
        elif sim.hover_status == "SP":
            fit = 0 - 50*sim.input_cost + sim.alpha + 0.01/vol
        elif sim.hover_status == "N":
            fit = -10 + 0.01/vol

        hover_status.append(sim.hover_status)
        input_cost.append(sim.input_cost)
        volume.append(vol)
        alpha.append(sim.alpha)
        fitness.append(fit)

    # Order population
    population_ordered = zip(population, fitness, hover_status, num_props, input_cost, volume, alpha)
    population_ordered = sorted(population_ordered, key=lambda x: x[1], reverse=True)
    population, fitness, hover_status, num_props, input_cost, volume, alpha = zip(*population_ordered)

    population = list(population)
    fitness = list(fitness)

    if verbose > 0:
        print(f"Top {verbose} individuals:")
        print("=================================================================")
        print("|Drone\t|#Props\t|Hover\t|Cost\t|Vol\t|Alpha\t|Fitness\t|")
        print("=================================================================")

        for idx in range(verbose):
            if input_cost[idx] is not None:
                print(f"|{idx+1}\t|{num_props[idx]}\t|{hover_status[idx]}\t|{input_cost[idx]:.3f}\t|{volume[idx]:.4f}\t|{alpha[idx]:.3f}\t|{fitness[idx]:.2f}\t\t|")
            else:
                print(f"|{idx+1}\t|{num_props[idx]}\t|{hover_status[idx]}\t|{input_cost[idx]}\t|{volume[idx]:.4f}\t|{alpha[idx]}\t|{fitness[idx]:.2f}\t\t|")

        print("-----------------------------------------------------------------")

    return population, fitness


def crossover(parent1, parent2):
    c_point = np.random.randint(1,parent1.shape[0]-2)
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)

    child1[0:c_point] = parent1[0:c_point]
    child1[c_point:] = parent2[c_point:]

    child2[0:c_point] = parent2[0:c_point]
    child2[c_point:] = parent1[c_point:]

    return child1, child2

def mutate(child, sigma=0.05):
    mut = np.random.uniform(0,1)
    if mut < sigma:
        mut_val = np.random.uniform(-0.2, 0.2, size=child.shape[0])
        mask = np.random.randint(0, 2, size=child.shape[0])
        child = child + mask*mut_val
        child = np.clip(child, -1, 1)
    return child

def roulette(population, fitness):
    roulette_pop = population.copy()
    roulette_fit = fitness.copy()
    selections = []
    for _ in range(2):
        min_fitness = min(roulette_fit)
        shifted_fit = roulette_fit - min_fitness

        normalized_fitness = shifted_fit/sum(shifted_fit)
        
        cumulative_probabilities = np.cumsum(normalized_fitness)

        r = np.random.uniform(low=0, high=1)

        for i, prob in enumerate(cumulative_probabilities):
                if r < prob:
                    selected_individual = roulette_pop.pop(i)
                    _ = roulette_fit.pop(i)
                    selections.append(selected_individual)
                    break

    parent1, parent2 = selections
    return parent1, parent2