import numpy as np
import yaml
from swarmsim.Populations import Population
from typing import Optional


class LightSensitive_PTW(Population):
    '''
    TODO 
    A class that implements a Persistent Turining Walkers that respond to external light inputs

    Parameters
    -------
        config_path : str 
            Path of the configuration file  

    Attributes
    -------

        
        PARAMETERS

    Configuration file requirements
    ------

        

    Notes
    ------
        For more details on how initial conditions can be generated see the method get_initial_conditions of the class Population

    '''

    def __init__(self, config_path: str, name: str = None) -> None:

        super().__init__(config_path, name)

        self.u_old: Optional[np.ndarray] = None

        self.theta_s: Optional[np.ndarray] = None
        self.mu_s: Optional[np.ndarray] = None
        self.alpha_s: Optional[np.ndarray] = None
        self.beta_s: Optional[np.ndarray] = None
        self.gamma_s: Optional[np.ndarray] = None
        self.sigma_s: Optional[np.ndarray] = None

        self.theta_w: Optional[np.ndarray] = None
        self.mu_w: Optional[np.ndarray] = None
        self.alpha_w: Optional[np.ndarray] = None
        self.beta_w: Optional[np.ndarray] = None
        self.gamma_w: Optional[np.ndarray] = None
        self.sigma_w: Optional[np.ndarray] = None

        self.params_shapes = {
            'theta_s': (),
            'mu_s': (),
            'alpha_s': (),
            'beta_s': (),
            'gamma_s': (),
            'sigma_s': (),

            'theta_w': (),
            'mu_w': (),
            'alpha_w': (),
            'beta_w': (),
            'gamma_w': (),
            'sigma_w': ()
        }


        self.dt = self.config["dt"]

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.theta_s = self.params['theta_s']
        self.mu_s = self.params['mu_s']
        self.alpha_s = self.params['alpha_s']
        self.beta_s = self.params['beta_s']
        self.gamma_s = self.params['gamma_s']
        self.sigma_s = self.params['sigma_s']

        self.theta_w = self.params['theta_w']
        self.mu_w = self.params['mu_w']
        self.alpha_w = self.params['alpha_w']
        self.beta_w = self.params['beta_w']
        self.gamma_w = self.params['gamma_w']
        self.sigma_w = self.params['sigma_w']

        self.u_old = np.zeros([self.N, self.input_dim])  # Initialization of the last control input applied

    def get_drift(self) -> np.array :
        '''
        Movement at constant speed \mu (\mu_1, \mu_2,..., \mu_n).

        Returns:
        ------
            drift: numpy array (num_agents x dim_state)
                Dirft of the population

        '''

        # State Variables: x, y, v, theta, omega(w)

        
        #self.x[:,2] = np.clip(self.x[:,2], 0,None)  # Ensure speed is non-negative
        v = self.x[:,2]
        self.x[:,3] = np.mod(self.x[:,3], 2 * np.pi)  # Ensure theta is in [0, 2*pi]
        theta = self.x[:,3]
        w = self.x[:,4]

        du = (self.u - self.u_old)/self.dt
        du_pos = np.max(np.hstack((du,np.zeros(du.shape))),axis=1)
        du_neg = np.min(np.hstack((du,np.zeros(du.shape))),axis=1)
        self.u_old = self.u

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dv = self.theta_s * (self.mu_s - v) + self.alpha_s * self.u[:,0] + self.beta_s * du_pos + self.gamma_s * du_neg
        dth = w
        dw = self.theta_w * (self.mu_w - w) + np.sign(w) * (self.alpha_w * self.u[:,0] + self.beta_w * du_pos + self.gamma_w * du_neg)


        drift = np.hstack((dx[:,np.newaxis],dy[:,np.newaxis],dv[:,np.newaxis],dth[:,np.newaxis],dw[:,np.newaxis]))
        return drift

    def get_diffusion(self) -> np.array :
        '''
        Stochastic diffusion in the environment (Standard Weiner Process).

        Returns:
        --------
            diffusion: numpy array (num_agents x dim_state)
                Diffusion of the population

        '''

        dx = np.zeros([self.N,1])
        dy = np.zeros([self.N,1])
        dv = self.sigma_s
        dth = np.zeros([self.N,1])
        dw = self.sigma_w

        diffusion = np.hstack((dx,dy,dv[:,np.newaxis],dth,dw[:,np.newaxis]))

        return diffusion


