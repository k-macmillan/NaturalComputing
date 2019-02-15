import numpy as np
import matplotlib.pyplot as plt


class GA():
    def __init__(self, pool=10, generations=10, debug=False):
        self.pool = pool
        self.generations = generations
        self.debug = debug
        self.state = 'hc'
        self.max_iter = 30
        self.InitializeX()

    def reset(self):
        self.x = np.copy(self.og)
        self.InitializeT()

    def InitializeX(self):
        self.og = np.random.uniform(0, 1, size=(1, self.pool))
        self.reset()
        self.function = np.linspace(0, 1, 2000)
        self.function = self.EvalX(self.function)
        self.max = np.zeros(self.max_iter)
        self.max_fitness = np.zeros(self.max_iter)

    def InitializeT(self):
        if self.state == 'hc':
            self.t = 0
        else:
            self.t = 0.25

    def UpdateT(self):
        if self.state == 'hc':
            self.t += 1
        else:
            self.t *= 0.8

    def Eval(self):
        self.fitness = np.power(2, -2 * ((self.x - 0.1) / 0.9)**2) * (np.sin(5 * np.pi * self.x)**6)
        # Make self.X a 2D array [x, fitness]
        self.X = np.concatenate((self.x, self.fitness), axis=0)
        # Sort and flip so the fitness is in order
        self.X = np.flip(np.argsort(self.X, axis=1))
        if self.debug:
            print('iteration: {}: {}\n'.format(self.t, self.fitness[0][self.X[0][0]]))
        # Note:
        # self.X is now in [fitness, x] order

    def EvalX(self, x):
        return np.power(2, -2 * ((x - 0.1) / 0.9)**2) * (np.sin(5 * np.pi * x)**6)

    def EvalXSA(self, x):
        ret_val = np.power(2, -2 * ((x - 0.1) / 0.9)**2) * (np.sin(5 * np.pi * x)**6)
        if ret_val > 1 or ret_val < 0:
            return -1000
        else:
            return ret_val

    def EvalSA(self):
        self.fitness = np.power(2, -2 * ((self.x - 0.1) / 0.9)**2) * (np.sin(5 * np.pi * self.x)**6)
        self.max_index = np.argsort(-self.fitness)[0][0]

    def Draw(self):        
        plt.scatter(self.x, self.fitness, color='r', marker='o')
        plt.plot(np.linspace(0, 1, 2000), self.function, color='b')
        plt.show()

    def DrawMax(self):
        plt.scatter(self.max, self.max_fitness, color='r', marker='o')
        plt.plot(np.linspace(0, 1, 2000), self.function, color='b')
        plt.show()

    def SetMax(self):
        self.max[self.t] = self.x[0][self.X[0][0]]
        self.max_fitness[self.t] = self.fitness[0][self.X[0][0]]

    def SetMaxSA(self):
        self.max[self.iteration] = self.x[0][self.max_index]
        self.max_fitness[self.iteration] = self.fitness[0][self.max_index]

    def MainLoopHC(self):
        self.Eval()
        self.SetMax()
        self.Draw()
        self.UpdateT()
        while (self.t < self.max_iter):
            self.PerturbX()
            self.Eval()
            self.SetMax()
            self.UpdateT()
        self.X = np.flip(self.X)
        self.Draw()
        self.DrawMax()
        print('Best x: ', self.x[0][self.X[0][0]])
        
    def MainLoopSA(self):
        self.state = 'sa'
        self.reset()
        self.iteration = 0
        self.EvalSA()
        self.SetMaxSA()
        self.Draw()
        while (self.iteration < self.max_iter):
            self.PerturbX()
            self.EvalSA()
            self.SetMaxSA()
            self.UpdateT()
            self.iteration += 1
        print('Best x: ', self.x[0][self.max_index])
        self.Draw()
        self.DrawMax()

    def PerturbX(self):
        count = 0
        for i in range(self.pool):
            modifier = np.abs(np.random.normal(0, 0.25, 1) * self.x[0][i] + self.x[0][i])
            fitness = self.EvalXSA(modifier)
            # If the fitness of the modified value > x, x = modified
            if self.fitness[0][i] < fitness:
                self.x[0][i] = modifier
                count += 1
            elif self.state == 'sa' and (np.random.random_sample() < np.exp(-np.abs(self.fitness[0][i] - fitness) / self.t)):
                self.x[0][i] = modifier
                # 10 to differentiate
                count += 10
        if self.debug:
            print('{}: improved perturbations: {}'.format(self.iteration, count))

    def Run(self):
        self.MainLoopHC()
        self.MainLoopSA()


if __name__ == '__main__':
    ga = GA()
    ga.Run()