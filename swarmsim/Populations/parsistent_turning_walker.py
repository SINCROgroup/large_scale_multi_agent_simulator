import numpy as np
import yaml
from swarmsim.Populations import Populations
from swarmsim.Utils import broadcast_parameter
from typing import Optional


class LightSensitive_PTW(Populations):
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

    def __init__(self, config_path:str) -> None:

        self.u_old: Optional[np.ndarray] = None

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

        super().__init__(config_path)

        self.dt = self.config["dt"]


    def get_drift(self) -> np.array :
        '''
        Movement at constant speed \mu (\mu_1, \mu_2,..., \mu_n).

        Returns:
        ------
            drift: numpy array (num_agents x dim_state)         
                Dirft of the population

        '''

        # State Variables: x, y, v, theta, omega(w)

        v = self.x[:,2]
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

    def reset(self) -> None:
        """
        Resets the state of the population to its initial conditions.

        This method reinitializes the agent states, external forces, and control inputs.
        """
        super().reset()

        self.theta_s = broadcast_parameter(self.params['theta_s'], ())
        self.mu_s = broadcast_parameter(self.params['mu_s'], ())
        self.alpha_s = broadcast_parameter(self.params['alpha_s'], ())
        self.beta_s = broadcast_parameter(self.params['beta_s'], ())
        self.gamma_s = broadcast_parameter(self.params['gamma_s'], ())
        self.sigma_s = broadcast_parameter(self.params['sigma_s'], ())

        self.theta_w = broadcast_parameter(self.params['theta_w'], ())
        self.mu_w = broadcast_parameter(self.params['mu_w'], ())
        self.alpha_w = broadcast_parameter(self.params['alpha_w'], ())
        self.beta_w = broadcast_parameter(self.params['beta_w'], ())
        self.gamma_w = broadcast_parameter(self.params['gamma_w'], ())
        self.sigma_w = broadcast_parameter(self.params['sigma_w'], ())

        self.u_old = np.zeros([self.N, self.input_dim])  # Initialization of the last control input applied

