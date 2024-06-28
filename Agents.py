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

    def compute_int(self):
        pass


class Herder(Population):
    def __init__(self, x0, v_max=10, sensing_radius=10000, name='herder'):
        super().__init__(x0=x0, param={'v_max': v_max, 'sensing_rad': sensing_radius}, name=name)
        self.v_max = self.param['v_max']
        self.sensing_rad = self.param['sensing_rad']

    def updater(self, u, interaction):
        v = np.clip(u + interaction, -self.v_max, self.v_max)
        self.x += v * self.dt

    def get_local_obs(self, state):
        loc_idx = np.where(np.linalg.norm(state, axis=1) < self.sensing_rad, state)
        return state[:, loc_idx]  # Check axis


class Target(Population):
    def __init__(self, x0, v_max=10, sigma=1, name='target'):
        super().__init__(x0=x0, param={'v_max': v_max, 'sigma:': sigma}, name=name)
        self.v_max = self.param['v_max']
        self.sigma = self.param['sigma']

    def updater(self, u, interaction):
        dx = np.clip(self.sigma * np.sqrt(self.dt) * np.random.normal() + interaction * self.dt, -self.v_max*self.dt, self.v_max*self.dt)
        self.x += dx

    def get_local_obs(self, state):
        return self.x


class HerderTargetInteraction(Interaction):
    def __init__(self, herder, target, param, adj):
        super().__init__(herder, target, param, adj)
        self.ht_distance = None
        self.hh_distance = None
        self.phase = None
        self.n_agents = [self.interacting_pop[i].n for i in range(len(self.interacting_pop))]
        self.c = self.param['c']
        self.gamma = self.param['gamma']
        self.lamb = self.param['lamb']

    def compute_int(self):
        self.ht_distance = scipy.spatial.distance_matrix(self.interacting_pop[0], self.interacting_pop[1])
        # self.hh_distance = scipy.spatial.distance_matrix(self.interacting_pop[0], self.interacting_pop[0])
        dataset1 = np.array(self.interacting_pop[0])
        dataset2 = np.array(self.interacting_pop[1])
        vectors = dataset2[:, np.newaxis, :] - dataset1
        angles = np.arctan2(vectors[:, :, 1], vectors[:, :, 0])
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        angles = angles.T
        self.phase = angles
        x = np.zeros((self.n_agents[0], 2))
        x[:, 0] = np.sum(self.c * 0.5 * (1 - np.tanh(self.gamma * (self.ht_distance - self.lamb) / self.lamb)) * np.cos(self.phase), axis=1)
        x[:, 1] = np.sum(self.c * 0.5 * (1 - np.tanh(self.gamma * (self.ht_distance - self.lamb) / self.lamb)) * np.sin(self.phase), axis=1)
        return x
