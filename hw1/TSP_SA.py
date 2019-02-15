import numpy as np
import matplotlib.pyplot as plt
import random
import math

#  Computer the tour length
def evaluate(cities):
    distance = 0
    for index in range(len(cities)):
        a = cities[index]
        if index == len(cities) - 1:
            b = cities[0]
        else:
            b = cities[index + 1]

        distance += np.linalg.norm(a - b)
        index += 1

    return distance

# A perturbation is a city swap
def swap_rand(x):
    i = random.randint(0, len(x) - 2)
    j = random.randint(i, len(x) - 1)

    y = np.copy(x)
    # swap cities and invert sublist
    y[i: j] = y[i: j][::-1]

    return y

def swap(x, gen, epoch):
    i = (gen * 2) % 50
    j = (gen * 2 + 1 + epoch) % 50
    y = np.copy(x)
    temp = np.copy(y[i])
    y[i] = np.copy(y[j])
    y[j] = np.copy(temp)
    y[i: j] = y[i: j][::-1]
    return y


def accept_solution(energy1, energy2, temperature):
    if energy1 > energy2:
        return True
    else:
        a = math.exp((energy1 - energy2) / temperature)
        b = random.random()
        if a > b:
            return True
        else:
            return False


def run(cities, cities_number, temperature = 800, cooling_factor = .001):
    current = evaluate(cities)
    first = current
    i = 0
    gen = 0
    epoch = 0
    while temperature > 0.001:
        # if gen == 0:
            # print('FIRST:')
            # print('Generation: ', gen)
            # print('Fitness: ', current)
            # print()
        # new_solution = swap(cities, gen, epoch)
        new_solution = swap_rand(cities)
        energy = evaluate(new_solution)
        if accept_solution(current, energy, temperature):
            cities = new_solution
            current = energy        

        # if (i%50==0):
            # plot(cities,path = 1, wait = 0)
        if gen == 14901:
            # print('LAST:')
            # print('Generation: ', gen)
            # print('Fitness: ', current)
            # print()

            print('{}\t{}'.format(first, current))
        if i % 50 == 0:
            plot(cities,path = 1, wait = 0)

        temperature *= 1 - cooling_factor
        i = i+1
        gen += 1
        if gen % 50 - 2 == 0:
            epoch += 1
    return cities


def plot(cities, path, wait):
    plt.clf()
    if (path == 1):
        plt.plot(cities[:, 0], cities[:, 1], color='red', zorder=0)
    plt.scatter(cities[:, 0], cities[:, 1], marker='o')
    plt.axis('off')
    if (wait == 0):  plt.ion()
    plt.show()
    plt.pause(.001)


print()
cities_number = 50
# for i in range(10, 50):
seed = 2
random.seed(seed)
np.random.seed(seed)
cities = (np.random.rand(cities_number, 2) * 100).astype(int)
plot(cities,path = 0, wait = 1)
cities = run(cities, cities_number, temperature = 3000)
plt.ioff()
plot(cities,path = 1, wait = 1)
# print('------------------------------------')

print()