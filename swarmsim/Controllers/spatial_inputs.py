from swarmsim.Controllers import Controller
import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from swarmsim.Utils import gaussian_input



class GaussianRepulsion(Controller):
    """
    This class implements a radial repulsion force whose intensity is shaped like a Gaussian distribution centered in the origin with 1 as standard deviation

    Arguments
    ---------
    population : Population
        The population where the control is exerted
    environment : Environment
        The environment where the agents live
    config_path: str
        The path of the configuration file
    
    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:    
    
    disc_pts: int
        Number of points of the discretization grid for the Gaussian input 
    
    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        GaussianRepulsion:
            disc_pts: 30

    This defines a `GaussianRepulsion` force that uses a 30x30 meshgrid to approximate the gaussian in the environment.

    """

    def __init__(self, population, environment, config_path=None) -> None:

        super().__init__(population, environment, config_path)


        x_dim = environment.dimensions[0]
        y_dim = environment.dimensions[1]
        
        self.disc_pts = self.params.get('disc_pts', 30)

        # Create a grid of points
        x = np.linspace(-x_dim, x_dim, self.disc_pts)  # disc_pts points between the bounds of the arena for x-axis
        y = np.linspace(-y_dim, y_dim, self.disc_pts)  # disc_pts points between the bounds of the arena for y-axis

        

        # Create the 2D grid of values
        X, Y = np.meshgrid(x, y)  # Create a grid from x and y

        Z = gaussian_input(np.transpose(X), np.transpose(Y),sigma_x=20.0,sigma_y=10.0)  # Apply the Gaussian function on the grid

        # Create the RegularGridInterpolator
        self.interpolator = sp.RegularGridInterpolator((x, y), np.transpose(Z), method='linear')   


        
    def get_action(self):
        """ 

            This method computes a radial repulsion force from the origin, whose intensity is scaled using 
            a Gaussian distribution in a 2D space.

        """

        rep_strength = self.interpolator(self.population.x) #Strength of repulsion from the center (Gaussian)
        dist = np.maximum(0.0001,(np.linalg.norm(self.population.x, axis=1)))  #Distances of the agents from the origin (Nx1) 
        rep_dir = (self.population.x)/dist[:,np.newaxis]    #Versor of the position of the agent (Nx2)
        return rep_strength[:,np.newaxis]*rep_dir           #Repulsion strength (Nx2)



    #Utility Function that defines a Gaussian distribution in a 2D Spce


class LightPattern(Controller):

    """
    This class implements a radial repulsion force whose intensity is shaped like a Gaussian distribution centered in the origin with 1 as standard deviation

    Arguments
    ---------
    population : Population
        The population where the control is exerted
    environment : Environment
        The environment where the agents live
    config_path: str
        The path of the configuration file
    
    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:    
    
    pattern_path: str
        The path (relative or absolute) of the pattern of light that you want to project in the environment
    
    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        LightPattern:
            pattern_path: ../Configuration/Config_data/BCL.jpeg

    This defines a `LightPattern` that project the content of BCL.jpeg over the environment.

    """

    def __init__(self, population, environment, config_path=None) -> None:

        super().__init__(population, environment, config_path)

        pattern_path = self.config.get("pattern_path","")
        camera_pattern = plt.imread(pattern_path)

        x_dim = environment.dimensions[0]
        y_dim = environment.dimensions[1]
        self.environment = environment

        # Create a grid of points
        x = np.linspace(-x_dim/2, x_dim/2, camera_pattern.shape[1])  
        y = np.linspace(-y_dim/2, y_dim/2, camera_pattern.shape[0])  

        Z = camera_pattern[:,:,2].T/255

        # Create the RegularGridInterpolator
        self.interpolator = sp.RegularGridInterpolator((x, y), Z, method='linear')   


    def get_action(self):
        
        """

            This function returns the light intensity projected by the pattern defined at the pattern_path. 
            All agents that are outside the boundaries of the arena are considered as being at the borders of the arena.
        
        """
        
        # All agents outside the bounds are considered at the bound
        limit_x = self.environment.dimensions[0]/2 * np.ones([self.population.x.shape[0],1])
        limit_y = self.environment.dimensions[1]/2 * np.ones([self.population.x.shape[0],1])
        limits_p = np.hstack((limit_x,limit_y))
        limits_n = -limits_p

        state = np.concatenate((self.population.x[:,[0,1],np.newaxis],limits_p[:,:,np.newaxis]),axis=2)
        state = np.min(state,axis=2)
        state = np.concatenate((state[:,:,np.newaxis],limits_n[:,:,np.newaxis]),axis=2)
        state = np.max(state,axis=2)
        
        #Get the light intensity at the position of the agents 
        light_ity = self.interpolator(state) 

        return light_ity[:,np.newaxis]
    

    def get_action_in_space(self,positions):

        """

            Returns the light intensity at the position specified in the positions vector

            Arguments
            ---------
            positions : numpy.Array(num_positions, num_dimensions)
                The positions where you want to retrieve the value of the control action

        """

        light_ity = self.interpolator(positions) 

        return light_ity[:,np.newaxis]


class Temporal_pulses(Controller):

    """
    This class implements a radial repulsion force whose intensity is shaped like a Gaussian distribution centered in the origin with 1 as standard deviation

    Arguments
    ---------
    population : Population
        The population where the control is exerted
    environment : Environment
        The environment where the agents live
    config_path: str
        The path of the configuration file
    
    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:    
    
    pattern_path: str
        The path (relative or absolute) of the pattern of light that you want to project in the environment
    
    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        LightPattern:
            pattern_path: ../Configuration/Config_data/BCL.jpeg

    This defines a `LightPattern` that project the content of BCL.jpeg over the environment.

    """

    def __init__(self, population, environment, config_path=None) -> None:

        super().__init__(population, environment, config_path)
        self.T = self.config.get('Period', 1.0)  # Period of the temporal pulses
        self.current_time = 0.0  # Initialize the current time

        


    def get_action(self):
        
        """

            This function returns the light intensity projected by the pattern defined at the pattern_path. 
            All agents that are outside the boundaries of the arena are considered as being at the borders of the arena.
        
        """
        self.current_time += self.dt  # Update the current time
        if np.mod(self.current_time,self.T) < (self.T/2):
            #print("t:"+ str(self.current_time)  +"YES light")
            u = np.ones((self.population.x.shape[0], 1))
        else:
            #print("t:"+ str(self.current_time)  +"NO light")
            u = np.zeros((self.population.x.shape[0], 1))
        
        

        return u
    

    def get_action_in_space(self,positions):

        """

            Returns the light intensity at the position specified in the positions vector

            Arguments
            ---------
            positions : numpy.Array(num_positions, num_dimensions)
                The positions where you want to retrieve the value of the control action

        """
        print("t:"+ str(np.mod(self.current_time,self.T)))
        if np.mod(self.current_time,self.T) < (self.T/2):
            u = np.ones((len(positions), 1))
        else:
            u = np.zeros((len(positions), 1))

        return u

 
class RectangularFeedback(Controller):
 
    """
    This class implements a radial repulsion force whose intensity is shaped like a Gaussian distribution centered in the origin with 1 as standard deviation
 
    Arguments
    ---------
    population : Population
        The population where the control is exerted
    environment : Environment
        The environment where the agents live
    config_path: str
        The path of the configuration file
   
    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:    
   
    pattern_path: str
        The path (relative or absolute) of the pattern of light that you want to project in the environment
   
    Examples
    --------
    Example YAML configuration:
 
    .. code-block:: yaml
 
        LightPattern:
            pattern_path: ../Configuration/Config_data/BCL.jpeg
 
    This defines a `LightPattern` that project the content of BCL.jpeg over the environment.
 
    """
 
    def __init__(self, population, environment, config_path=None) -> None:
 
        super().__init__(population, environment, config_path)
 
 
        x_dim = environment.dimensions[0]
        y_dim = environment.dimensions[1]
        self.environment = environment
   
        # Create a grid of points
        self.x = np.linspace(-x_dim/2, x_dim/2, self.environment.dimensions[0])  
        self.y = np.linspace(-y_dim/2, y_dim/2, self.environment.dimensions[1])  
       
        self.update=50
        self.count = self.update
        img = Image.new('RGB', (self.environment.dimensions[0], self.environment.dimensions[1]), (0, 0, 0))
        light_pattern = np.array(img)
        Z = light_pattern[:,:,2].T/255
        self.interpolator = sp.RegularGridInterpolator((self.x, self.y), Z, method='linear')
 
 
    def get_action(self):
       
        """
 
            This function returns the light intensity projected by the pattern defined at the pattern_path.
            All agents that are outside the boundaries of the arena are considered as being at the borders of the arena.
       
        """
        # All agents outside the bounds are considered at the bound
        limit_x = self.environment.dimensions[0]/2 * np.ones([self.population.x.shape[0],1])
        limit_y = self.environment.dimensions[1]/2 * np.ones([self.population.x.shape[0],1])
        limits_p = np.hstack((limit_x,limit_y))
        limits_n = -limits_p
 
        state = np.concatenate((self.population.x[:,[0,1],np.newaxis],limits_p[:,:,np.newaxis]),axis=2)
        state = np.min(state,axis=2)
        state = np.concatenate((state[:,:,np.newaxis],limits_n[:,:,np.newaxis]),axis=2)
        state = np.max(state,axis=2)

        ang_diff = np.abs(self.population.x[:,3] - np.atan2(self.population.x[:,1], self.population.x[:,0]))
        mask = np.double(ang_diff < np.pi/2)
        #print("Mask: ", mask)
        light_ity = mask


        return light_ity[:,np.newaxis]
 
 
    def get_action_in_space(self,positions):
 
        """
 
            Returns the light intensity at the position specified in the positions vector
 
            Arguments
            ---------
            positions : numpy.Array(num_positions, num_dimensions)
                The positions where you want to retrieve the value of the control action
 
        """
 
        light_ity = self.interpolator(positions)
        print("Max light intensity: ", np.max(light_ity))
 
        return light_ity[:,np.newaxis]


class AngularFeedback(Controller):

    def __init__(self, population, environment, config_path=None) -> None:

        

        super().__init__(population, environment, config_path)

        
        

        # Set the parameters of the controller
        self.line_width = self.config.get('line_width', 3)  # Width of the lines drawn
        self.angle_width = self.config.get('angle_width', 0.5*45*np.pi/180)  # Width of the angle in radians
        self.agent_distance = self.config.get('agent_distance', 2*5)  # Distance behind
        self.update_time = self.config.get('update_time', 20*self.dt)  # Change in update time (otherwise it depends on the simulation dt)
        self.last_update = -self.update_time  # Initialize the last update time
        self.current_time = 0  # Update the current time



        # Initialize the controller with a black Image
        x_dim = environment.dimensions[0]
        y_dim = environment.dimensions[1]
        self.environment = environment
        # Create a grid of points
        self.x = np.linspace(-x_dim/2, x_dim/2, self.environment.dimensions[0])  
        self.y = np.linspace(-y_dim/2, y_dim/2, self.environment.dimensions[1])  
        img = Image.new('RGB', (self.environment.dimensions[0], self.environment.dimensions[1]), (0, 0, 0))
        light_pattern = np.array(img)
        Z = light_pattern[:,:,2].T/255
        self.interpolator = sp.RegularGridInterpolator((self.x, self.y), Z, method='linear')
        self.img_count = 0




    def get_action(self):
        
        """
            TBD
        """

        state = self.population.x[:,[0,1]]           # Get the position of all the agents
        state = np.clip(state, [-self.environment.dimensions[0]/2, -self.environment.dimensions[1]/2], [self.environment.dimensions[0]/2, self.environment.dimensions[1]/2])  # Clip the state to the limits of the environment

        self.current_time += self.dt  # Update the current time


        if (self.current_time - self.last_update >= self.update_time):

            self.last_update = self.current_time  # Update the last update time

            phi = np.arctan2(state[:,1], state[:,0]) # Calculate angle with respect to origin
            r = np.linalg.norm(state, axis=1)  # Calculate the distance from the origin

        
            img = Image.new('RGB', (self.environment.dimensions[0], self.environment.dimensions[1]), (0, 0, 0))
            img_draw = ImageDraw.Draw(img)


            x_start = (r+self.agent_distance)*np.cos(phi) + self.environment.dimensions[0] / 2
            y_start = (r+self.agent_distance)*np.sin(phi) + self.environment.dimensions[1] / 2

            x_incr = self.agent_distance * np.tan(self.angle_width) * np.sin(phi)
            y_incr = - self.agent_distance * np.tan(self.angle_width) * np.cos(phi)


            x_end1 = state[:,0] + x_incr + self.environment.dimensions[0] / 2
            y_end1 = state[:,1] + y_incr + self.environment.dimensions[1] / 2

            x_end2 = state[:,0] - x_incr + self.environment.dimensions[0] / 2
            y_end2 = state[:,1] - y_incr + self.environment.dimensions[1] / 2
            
            

            #x_end1 = r * np.cos(phi + self.angle_width) + self.environment.dimensions[0] / 2
            #y_end1 = r * np.sin(phi + self.angle_width) + self.environment.dimensions[1] / 2

            #x_end2 = r * np.cos(phi - self.angle_width) + self.environment.dimensions[0] / 2
            #y_end2 = r * np.sin(phi - self.angle_width) + self.environment.dimensions[1] / 2


            for i in range(0,len(state[:,1])):
                img_draw.line([x_start[i], y_start[i], x_end1[i], y_end1[i]], fill=(0,0,255), width=self.line_width)  # blue line for the center
                img_draw.line([x_start[i], y_start[i], x_end2[i], y_end2[i]], fill=(0,0,255), width=self.line_width)  # blue line for the center


            self.img_count += 1
            light_pattern = np.array(img)
            Z = light_pattern[:,:,2].T/255  # Extract the blue channel and transpose it
            self.interpolator = sp.RegularGridInterpolator((self.x, self.y), Z, method='linear')

            #img.save("logs/" + str(self.img_count) + ".png")  # Save the image for debugging



        light_ity = self.interpolator(state)
        return light_ity[:,np.newaxis]




    def get_action_in_space(self,positions):

        """

            Returns the light intensity at the position specified in the positions vector

            Arguments
            ---------
            positions : numpy.Array(num_positions, num_dimensions)
                The positions where you want to retrieve the value of the control action

        """

        light_ity = self.interpolator(positions) 

        return light_ity[:,np.newaxis]