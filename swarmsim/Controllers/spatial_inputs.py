from swarmsim.Controllers import Controller
import numpy as np
import scipy.interpolate as sp


class GaussianRepulsion(Controller):
    """
    This class implements a radial repulsion force whose intensity goes like a gaussion Gaussian repulsion 

    Arguments
    ---
    disc_pts (int): Number of points of the discretization grid for the Gaussian input 
    ity (float): Max intensity of the force

    """

    def __init__(self, population, environment, config_path=None) -> None:
        """ 
            This method initializes the repulsion force. It creates an interpolated Gaussian in 2D and 
            initializes the population that the force will act upon, the environment and loads the 
            configuration parameters
        """
        super().__init__(population, environment, config_path)


        x_dim = environment.dimensions[0]
        y_dim = environment.dimensions[1]
        
        self.disc_pts = self.params.get('disc_pts', 30)

        # Create a grid of points
        x = np.linspace(-x_dim/2, x_dim/2, self.disc_pts)  # disc_pts points between the bounds of the arena for x-axis
        y = np.linspace(-y_dim/2, y_dim/2, self.disc_pts)  # disc_pts points between the bounds of the arena for y-axis

        

        # Create the 2D grid of values
        X, Y = np.meshgrid(x, y)  # Create a grid from x and y

        Z = self.gaussian_input(np.transpose(X), np.transpose(Y),sigma_x=20.0,sigma_y=10.0)  # Apply the Gaussian function on the grid

        # Create the RegularGridInterpolator
        self.interpolator = sp.RegularGridInterpolator((x, y), np.transpose(Z), method='linear')   


        
    def get_action(self):
        """ 
            This method computes a radial repulsion force from the origin, whose intensity is scaled using 
            a Gaussian distribution in a 2D space 
        """

        rep_strength = self.interpolator(self.population.x) #Strength of repulsion from the center (Gaussian)
        dist = (np.linalg.norm(self.population.x, axis=1))  #Distances of the agents from the origin (Nx1) 
        rep_dir = (self.population.x)/dist[:,np.newaxis]    #Versor of the position of the agent (Nx2)
        return rep_strength[:,np.newaxis]*rep_dir           #Repulsion strength (Nx2)



    #Utility Function that defines a Gaussian distribution in a 2D Spce
    def gaussian_input(self,x, y, A=5.0, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0):
        """
        This method, given the parametes of a Gaussin and some points where to evaluate the function,
        computes a 2D Gaussian distribution. Assumption: the covariance matrix is diagonal
        Arguments
        ---
        A (float):      Maximum amplitude of the signal
        mu_x(float):    Average on the first dimension
        mu_y(float):    Average on the second dimension
        sigma_x(float): Standard deviation in the first dimension
        sigma_y(float): Standard deviation in the second dimension
        """
        return A * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2)) - ((y - mu_y)**2 / (2 * sigma_y**2)))
