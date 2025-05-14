import numpy as np

def crossover(parent1, parent2):
    # Arithmetic crossover
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)

    for i in range(parent1.shape[0]):
        beta = np.random.uniform(low=0, high=1)
        child1[i] = beta*parent1[i] + (1-beta)*parent2[i]
        child2[i] = (1-beta)*parent1[i] + beta*parent2[i]

    return child1, child2

def mutate(child, sigma=0.2, mode="rollover"):
    mut = np.random.uniform(0,1)
    if mut <= sigma:
        mut_val = np.random.uniform(-0.2, 0.2, size=child.shape[0])
        mut_val[0] = np.random.uniform(-0.8, 0.8)

        child = child + mut_val

        if mode == "rollover":
            for i in range(len(child)):
                if child[i] > 1:
                    child[i] = child[i] - 2
                elif child[i] < -1:
                    child[i] = child[i] + 2

        elif mode == "clip":
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

def tournament(rank, distance):
    pop_size = len(rank)
    s1 = np.random.randint(low=0, high=pop_size)
    s2 = None
    while s2 == s1 or s2 == None:
        s2 = np.random.randint(low=0, high=pop_size)

    if rank[s1] < rank[s2]:
        return s1
    
    elif rank[s2] < rank[s1]:
        return s2
    
    elif rank[s1] == rank[s2]:
        if distance[s1] >= distance[s2]:
            return s1
        else:
            return s2
    