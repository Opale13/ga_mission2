# maxone.py
# Author: Sébastien Combéfis
# Version: April 26, 2020

import random
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from tabulate import tabulate

PROB_MATING = 0.5
PROB_MUTATION = 0.2

# Initialising the population.
population = []

# Defining the fitness function.
def evaluate(ind):
    return sum(ind)

# Defining the mating function.
def uniform_mate(ind1, ind2):
    assert len(ind1) == len(ind2), "ind1 & ind2 should be the same length"
    child1 = ind1.copy()
    child2 = ind2.copy()

    for i in range(len(ind1)):
        if random.random() >= PROB_MATING:
            child1[i], child2[i] = child2[i], child1[i]

    return child1, child2

def cross_mate(ind1, ind2):
    assert len(ind1) == len(ind2), "ind1 & ind2 should be the same length"
    child1 = ind1.copy()
    child2 = ind2.copy()

    pivot = int(round(len(ind1)/2, 0))

    child1 = ind1[:pivot] + ind2[pivot:]
    child2 = ind1[pivot:] + ind2[:pivot]

    return child1, child2

# Defining the mutation function.
def mutate(ind):
    for i in range(len(ind)):
        if random.random() >= PROB_MUTATION:
            random_gene = random.randint(0, len(ind)-1)
            ind[i], ind[random_gene] = ind[random_gene], ind[i]
            break

    return ind

# Defining the selection function.
def select(pop):
    parents = list()

    all_fitness = [sum(individual) for individual in pop]

    for i in range(2):
        pop_sum = 0
        limit = random.randint(0, len(pop))
        
        for index, fitness_value in enumerate(all_fitness):
            pop_sum += fitness_value

            if pop_sum > limit:
                parents.append(pop[index])
                break

            if index == len(all_fitness)-1:
                parents.append(pop[index])

    return tuple(parents)

# GA
def main(pop_size: int, iterations: int, mate_function):
    IND_SIZE = 10
    POP_SIZE = pop_size
    ITERATIONS = iterations
    
    # Create initial population
    for i in range(POP_SIZE):
        population.append([random.randrange(2) for i in range(IND_SIZE)])

    stats = list()
    for i in range(ITERATIONS):
        parents = select(population)
        children = mate_function(parents[0], parents[1])
        mutate_children = list(map(mutate, children))

        # Remove worst individuals (2)
        for j in range(2):
            iteration_fitness = list(map(evaluate, population))
            del population[iteration_fitness.index(min(iteration_fitness))]

        # Add children to population
        for child in mutate_children:
            population.append(child)

        iteration_fitness = np.array(list(map(evaluate, population)))
        stats.append((i+1, np.mean(iteration_fitness), np.min(iteration_fitness), np.max(iteration_fitness)))
    
    return stats

# Running the simulation.
if __name__ == '__main__':
    ITERATIONS = 150
    
    # Parameters to change
    pop_size_list = [20, 50, 100, 150, 200]
    mate_function_list = [uniform_mate, cross_mate]

    # Utils for plot
    mate_function_subplot_list= [211, 212]
    plot_title_list = ["Mean value by iteration for uniform mate", "Mean value by iteration for cross mate"]
    linestyle_list = ['dashed', 'dashdot', 'dotted']

    stats_dict = dict()
    for pop_size in pop_size_list:
        if pop_size not in stats_dict:
            stats_dict[pop_size] = dict()

        for mate_function in mate_function_list:
            if mate_function.__name__ not in stats_dict[pop_size]:
                stats_dict[pop_size][mate_function.__name__] = list()

            stats = main(pop_size, ITERATIONS, mate_function)
            stats_dict[pop_size][mate_function.__name__].extend(stats)

    
    df = pd.DataFrame.from_dict(stats_dict)


    for pop_index, pop_size in enumerate(pop_size_list):
        for mate_index, mate_function in enumerate(mate_function_list):
            ax = plt.subplot(mate_function_subplot_list[mate_index])
            ax.plot(
                np.arange(ITERATIONS), 
                [stat[1] for stat in df[pop_size][mate_function.__name__]], 
                label='{}'.format(pop_size),
                linestyle=linestyle_list[pop_index%len(linestyle_list)])
    
            ax.legend()

    for index, mate_function_subplot in enumerate(mate_function_subplot_list):
        ax = plt.subplot(mate_function_subplot_list[index])
        ax.set_title(plot_title_list[index])
        ax.set_ylabel('Means')
        ax.set_xlabel('Iterations')

    plt.show()