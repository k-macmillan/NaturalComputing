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
    i = random.randint(0, x.size - 2)
    j = random.randint(i, x.size - 1)

    y = np.copy(x)
    # swap cities and invert sublist
    y[i: j] = y[i: j][::-1]

    return y

def swap(x, gen, epoch):
    i = (gen * 2) % (x.size // 2)
    j = (gen * 2 + 1 + epoch) % (x.size // 2)
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

def Reproduce(cities, det, gen, epoch):
    childbody = np.zeros((5, 40, 2), dtype=int)
    childleg = np.zeros((5, 10, 2), dtype=int)

    for i in range(5):
        bod_min = i * 10
        bod_max = bod_min + 10
        childbody[i] = np.concatenate([np.copy(cities[:bod_min, :]), np.copy(cities[bod_max:, :])], axis=0)
        # Take only the leg
        childleg[i] = np.copy(cities[bod_min:bod_max])

    cb, cl = Mutate(childbody, childleg, det, gen, epoch)
    y = Regrow(cb, cl, cities)
    return y

def Mutate(cb, cl, det, gen, epoch):
    if det:
        for i in range(5):
            cb[i] = swap(cb[i], gen, epoch)
            cl[i] = swap(cl[i], gen, epoch)
    else:
        for i in range(5):
            cb[i] = swap_rand(cb[i])
            cl[i] = swap_rand(cl[i])

    return cb, cl

def Regrow(cb, cl, x):
    y = np.zeros((10, 50, 2), dtype=np.int)
    for i in range(5):
        bod_min = i * 10
        bod_max = bod_min + 10
        y[i * 2] = np.copy(x)
        y[i * 2+ 1] = np.copy(x)
        # Body
        y[i, :bod_min, :] = cb[i, :bod_min, :]
        y[i, bod_max:, :] = cb[i, bod_min:, :]
        # Leg
        y[i + 1, bod_min:bod_max, :] = cl[i]

    return y


def run(cities, cities_number, temperature = 800, cooling_factor = .001):
    det = False
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
        y = Reproduce(cities, det, gen, epoch)
        max_energy = current
        min_index = -1
        for j in range(10):
            energy = evaluate(y[j])
            if accept_solution(current, energy, temperature):
                cities = np.copy(y[j])
                current = energy
                if max_energy > current:
                    max_energy = current
                    min_index = j

        # Overcomes energy mixups
        if min_index != -1:
            cities = np.copy(y[min_index])
            current = max_energy

        if (i%50==0):
            plot(cities,path = 1, wait = 0)
        if gen == 14901:
            # print('LAST:')
            # print('Generation: ', gen)
            # print('Fitness: ', current)
            # print()
            print('{}\t{}'.format(first, current))

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
