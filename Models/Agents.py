import numpy as np
import scipy


class Population:
    def __init__(self, x0, param, dt=0.03, name='p1'):
        self.x = x0  # Initial conditions
        self.param = param  # Parameters of the agents. This may be a vector of parameters for each agent in case of heterogeneity.
        self.n = np.size(x0, axis=0)  # Number of agents
        self.dt = dt  # Should come from simulator
        self.name = name  # e.g. herder, useful for logging and identifying agents
        self.ide = [self.name + str(i) for i in range(self.n)]  # ID of each agent of the population

    def updater(self, u, interaction):  # Defines step
        pass

    def get_local_obs(self, state):  # Get local observation of an agent of the population
        pass


class Interaction:
    def __init__(self, pop1, pop2, param, adj):
        self.interacting_pop = [pop1, pop2]  # List of the states of the two interacting populations
        self.param = param  # Parameters that characterize the interaction law
        self.adj = adj  # Adjacency matrix (if needed)

    def compute_int(self):  # Computes interaction term (to pass to the updater)
        pass


