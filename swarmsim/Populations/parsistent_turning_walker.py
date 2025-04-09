import numpy as np
import yaml
from swarmsim.Populations import Populations


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

        super().__init__(config_path)
        
        N = self.N
        self.id = self.config["id"]              # Population ID
        self.dt = self.config["dt"]



        self.f = np.zeros(self.x.shape)          # Initialization of the external forces
        self.u = np.zeros([self.N,1])            # Initialization of the control input
        self.u_old = np.zeros([self.N,1])        # Initialization of the last control input applied

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

        du = (self.u[:,0] - self.u_old[:,0])/self.dt
        du_pos = np.max(du[np.newaxis,:]+np.zeros(du.shape)[:,np.newaxis],axis=1)
        du_neg = np.min(du[np.newaxis,:]+np.zeros(du.shape)[:,np.newaxis],axis=1)
        self.u_old = self.u

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dv = self.params["theta_s"].values * (self.params["mu_s"].values - v) + self.params["alpha_s"].values * self.u[:,0] + self.params["beta_s"].values * du_pos + self.params["gamma_s"].values * du_neg
        dth = w
        dw = self.params["theta_w"].values * (self.params["mu_w"].values - w) + np.sign(w) * (self.params["alpha_w"].values * self.u[:,0] + self.params["beta_w"].values * du_pos + self.params["gamma_w"].values * du_neg)

        drift = np.hstack((dx[:,np.newaxis],dy[:,np.newaxis],dv[:,np.newaxis],dth[:,np.newaxis],dw[:,np.newaxis]))
        return drift

    def get_diffusion(self) -> np.array :
        '''
        Stochastic diffusion in the environment (Standard Weiner Process).

        Returns:
        ------
            diffusion: numpy array (num_agents x dim_state)
                Diffusion of the population

        '''

        dx = np.zeros([self.N,1])
        dy = np.zeros([self.N,1])
        dv = self.params["sigma_s"].values
        dth = np.zeros([self.N,1])
        dw = self.params["sigma_w"].values

        diffusion = np.hstack((dx,dy,dv[:,np.newaxis],dth,dw[:,np.newaxis]))

        return diffusion

    def reset_state(self) -> None:
        '''
        
        Resets the state to the initial conditions.

        '''
        
        self.get_initial_conditions()
        self.f = np.zeros(self.x.shape)  # Initialization of the external forces
        self.u = np.zeros(self.x.shape)  # Initialization of the control input

        