from swarmsim.Environments import Environment
from swarmsim.Controllers import Controller
import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from swarmsim.Populations.population import Population
from swarmsim.Utils import gaussian_input



class GaussianRepulsion(Controller):
    """
    Implements a spatially-varying radial repulsion force with Gaussian intensity profile.

    This controller creates a repulsion field centered at the origin with intensity that
    follows a 2D Gaussian distribution. Agents experience repulsive forces that push them
    away from the center, with force magnitude determined by the Gaussian profile and
    direction determined by their position relative to the origin.

    The controller uses spatial interpolation to efficiently compute force values at
    agent positions from a discretized Gaussian field defined on a regular grid.

    Parameters
    ----------
    population : Population
        The population of agents to be controlled by this repulsion field.
    environment : Environment
        The environment providing spatial boundaries and dimensions.
    config_path : str, optional
        Path to the YAML configuration file containing controller parameters. Default is None.

    Attributes
    ----------
    disc_pts : int
        Number of discretization points along each axis for the Gaussian field grid.
    interpolator : scipy.interpolate.RegularGridInterpolator
        Interpolator object for efficient evaluation of the Gaussian field at arbitrary positions.

    Config Requirements
    -------------------
    The YAML configuration file should contain the following parameters:

    dt : float
        Sampling time of the controller.
    disc_pts : int, optional
        Number of points in the discretization grid for the Gaussian input field.
        Default is ``30``.

    Notes
    -----
    The repulsion force is computed as:

        F(x, y) = G(x, y) * (x, y) / ||x, y||

    where:
    - G(x, y) is the Gaussian intensity at position (x, y)
    - (x, y) / ||x, y|| is the unit vector pointing away from the origin
    - The result is a radial repulsion force with Gaussian-modulated strength

    The Gaussian field is pre-computed on a regular grid and interpolated for efficiency.
    The grid spans the environment dimensions with the specified number of discretization points.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        GaussianRepulsion:
            dt: 0.1
            disc_pts: 50



    """

    def __init__(self, population: Population, environment: Environment, config_path=None) -> None:
        """
        Initialize the Gaussian repulsion controller.

        Parameters
        ----------
        population : Population
            The population to be controlled by the repulsion field.
        environment : Environment
            The environment providing spatial dimensions.
        config_path : str, optional
            Path to the configuration file. Default is None.
        """

        super().__init__(population, environment, config_path)


        x_dim = environment.dimensions[0]
        y_dim = environment.dimensions[1]
        
        self.disc_pts = self.config.get('disc_pts', 30)

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
        Compute the Gaussian repulsion forces for all agents.

        This method evaluates the Gaussian intensity field at each agent's position
        and computes radial repulsion forces pointing away from the origin. The force
        magnitude is modulated by the Gaussian intensity profile.

        Returns
        -------
        np.ndarray
            Repulsion forces of shape (N, 2) where N is the number of agents.
            Each row contains the [x, y] force components for the corresponding agent.

        Notes
        -----
        The repulsion force computation involves:

        1. **Intensity Evaluation**: Sample Gaussian field at agent positions using interpolation
        2. **Distance Calculation**: Compute distance from each agent to the origin
        3. **Direction Calculation**: Compute unit vectors pointing away from origin
        4. **Force Combination**: Multiply intensity by direction to get final forces

        Agents at the exact origin (distance = 0) are assigned a small offset to avoid
        division by zero in the direction calculation.
        """

        rep_strength = self.interpolator(self.population.x) #Strength of repulsion from the center (Gaussian)
        dist = np.maximum(0.0001,(np.linalg.norm(self.population.x, axis=1)))  #Distances of the agents from the origin (Nx1) 
        rep_dir = (self.population.x)/dist[:,np.newaxis]    #Versor of the position of the agent (Nx2)
        return rep_strength[:,np.newaxis]*rep_dir           #Repulsion strength (Nx2)



    #Utility Function that defines a Gaussian distribution in a 2D Spce


class LightPattern(Controller):
    """
    Projects a spatial light pattern loaded from an image file onto the environment.

    This controller reads an image file and uses it to create a spatial control field
    that agents can sense. The image is mapped to the environment coordinates, and
    the blue channel intensity is used as the control signal. This enables complex
    spatial patterns to be used for agent guidance and control.

    The controller is particularly useful for biological agent models where light
    intensity can influence agent behavior, such as phototaxis or photophobic responses.

    Parameters
    ----------
    population : Population
        The population of agents that will sense the light pattern.
    environment : Environment
        The environment providing spatial boundaries and coordinate mapping.
    config_path : str, optional
        Path to the YAML configuration file containing controller parameters. Default is None.

    Attributes
    ----------
    interpolator : scipy.interpolate.RegularGridInterpolator
        Interpolator object for evaluating light intensity at arbitrary positions.
    environment : Environment
        Reference to the environment for boundary checking.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters:

    dt : float
        Sampling time of the controller.
    pattern_path : str
        Path (relative or absolute) to the image file containing the light pattern.
        Supported formats include JPEG, PNG, and other common image formats.

    Notes
    -----
    The controller implementation:

    1. **Image Loading**: Reads the specified image file using matplotlib
    2. **Coordinate Mapping**: Maps image pixels to environment coordinates
    3. **Channel Extraction**: Uses the blue channel (RGB[2]) as intensity values
    4. **Normalization**: Normalizes pixel values to [0, 1] range
    5. **Interpolation**: Creates interpolator for continuous intensity evaluation

    Agents outside the environment boundaries are clamped to the boundary values
    to ensure consistent behavior at the edges.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        LightPattern:
            dt: 0.1
            pattern_path: ../Configuration/Config_data/BCL.jpeg

    """

    def __init__(self, population: Population, environment: Environment, config_path: str = None) -> None:
        """
        Initialize the light pattern controller.

        Parameters
        ----------
        population : Population
            The population that will sense the light pattern.
        environment : Environment
            The environment providing spatial coordinate mapping.
        config_path : str, optional
            Path to the configuration file. Default is None.

        Raises
        ------
        FileNotFoundError
            If the specified pattern image file cannot be found.
        KeyError
            If the required pattern_path parameter is missing from configuration.
        """

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
        Get the light intensity values at the current agent positions.

        This method evaluates the light pattern at each agent's current position,
        applying boundary clamping to ensure agents outside the environment
        boundaries receive the edge intensity values.

        Returns
        -------
        np.ndarray
            Light intensity values of shape (N, 1) where N is the number of agents.
            Values are normalized to the range [0, 1] based on the image data.

        Notes
        -----
        The method performs the following steps:

        1. **Boundary Clamping**: Clip agent positions to environment boundaries
        2. **Interpolation**: Evaluate light pattern at clamped positions
        3. **Reshaping**: Return as column vector for compatibility

        Agents outside the defined pattern area receive the boundary values.
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
        Evaluate the light pattern at specified spatial positions.

        This method allows querying the light intensity at arbitrary positions
        in the environment, useful for analysis and visualization.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (num_positions, 2) containing [x, y] coordinates
            where light intensity should be evaluated.

        Returns
        -------
        np.ndarray
            Light intensity values of shape (num_positions, 1) at the specified positions.
            Values are normalized to the range [0, 1] based on the image data.

        Notes
        -----
        This method provides direct access to the interpolated light pattern.

        Useful for:
        - Analyzing light patterns before simulation
        - Creating visualizations of the control field
        - Implementing predictive control strategies
        """

        light_ity = self.interpolator(positions) 

        return light_ity[:,np.newaxis]


class Temporal_pulses(Controller):
    """
    Implements a temporally-varying control signal with periodic on/off pulses.

    This controller generates a uniform control signal that alternates between
    'on' (value = 1) and 'off' (value = 0) states with a specified period.
    The signal is spatially uniform but varies in time, creating synchronized
    temporal stimulation across all agents.

    The controller is useful for studying temporal entrainment, circadian rhythm
    effects, or synchronized behavioral responses in agent populations.

    Parameters
    ----------
    population : Population
        The population of agents to receive the temporal pulse signal.
    Period: float, optional
        The period of the temporal pulses (time for one complete on/off cycle).
    config_path : str, optional
        Path to the YAML configuration file containing controller parameters. Default is None.

    Attributes
    ----------
    T : float
        Period of the temporal pulses (time for one complete on/off cycle).
    current_time : float
        Current simulation time, updated at each control action.

    Config Requirements
    -------------------
    The YAML configuration file should contain the following parameters:

    dt : float
        Sampling time of the controller.
    Period : float, optional
        Period of the temporal pulses in simulation time units. Default is ``1.0``.

    Notes
    -----
    The pulse pattern follows a square wave:
    - First half of period (0 ≤ t mod T < T/2): Signal = 1 (ON)
    - Second half of period (T/2 ≤ t mod T < T): Signal = 0 (OFF)

    The controller maintains its own internal time counter that is incremented
    by `dt` at each call to `get_action()`.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        Temporal_pulses:
            dt: 0.01
            Period: 2.0

    This creates pulses with 2-second period (1 second ON, 1 second OFF).


    """

    def __init__(self, population: Population, environment: Environment, config_path: str = None) -> None:
        """
        Initialize the temporal pulse controller.

        Parameters
        ----------
        population : Population
            The population to receive temporal pulse signals.
        environment : Environment
            The environment (required for interface compatibility).
        config_path : str, optional
            Path to the configuration file. Default is None.
        """

        super().__init__(population, environment, config_path)
        self.T = self.config.get('Period', 1.0)  # Period of the temporal pulses
        self.current_time = 0.0  # Initialize the current time

        


    def get_action(self):
        """
        Generate the current temporal pulse state for all agents.

        This method updates the internal time counter and returns the current
        pulse state (1 for ON, 0 for OFF) based on the temporal position
        within the pulse period.

        Returns
        -------
        np.ndarray
            Pulse state array of shape (N, 1) where N is the number of agents.
            All agents receive the same pulse value: 1 during ON phase, 0 during OFF phase.

        Notes
        -----
        The method performs the following:

        1. **Time Update**: Increment internal time by dt
        2. **Phase Calculation**: Compute position within current pulse period
        3. **State Determination**: Return 1 if in first half of period, 0 otherwise

        The pulse timing is deterministic and synchronized across all agents,
        making it suitable for studying collective temporal responses.
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


class AngularFeedback(Controller):
    """
    Implements dynamic visual feedback using angular light patterns behind agents.

    This controller creates dynamic light patterns by drawing lines behind each agent
    based on their current position and a specified angular width. The patterns are
    updated periodically and projected as visual feedback that agents can sense.
    
    The controller generates wedge-shaped light patterns extending radially outward
    from each agent's position, creating a dynamic feedback system that can influence
    agent behavior through phototaxis or photophobic responses.

    Parameters
    ----------
    population : Population
        The population of agents for which light patterns will be generated.
    environment : Environment
        The environment providing spatial boundaries and coordinate mapping.
    config_path : str, optional
        Path to the YAML configuration file containing controller parameters. Default is None.

    Attributes
    ----------
    line_width : int
        Width of the lines drawn in the light pattern (in pixels).
    angle_width : float
        Angular width of the wedge pattern behind each agent (in radians).
    agent_distance : float
        Distance behind each agent where the wedge pattern starts.
    update_time : float
        Time interval between pattern updates (in simulation time units).
    last_update : float
        Timestamp of the last pattern update.
    current_time : float
        Current simulation time.
    x : np.ndarray
        Grid x-coordinates for spatial discretization.
    y : np.ndarray
        Grid y-coordinates for spatial discretization.
    interpolator : scipy.interpolate.RegularGridInterpolator
        Interpolator for evaluating light intensity at arbitrary positions.
    img_count : int
        Counter for saved debug images.

    Config Requirements
    -------------------
    The YAML configuration file should contain the following parameters:

    dt : float
        Sampling time of the controller.
    line_width : int, optional
        Width of the drawn lines in pixels. Default is ``3``.
    angle_width : float, optional
        Angular width of the wedge pattern in radians. Default is ``0.5 * 45 * π/180`` (22.5 degrees).
    agent_distance : float, optional
        Distance behind agents where patterns start. Default is ``10``.
    update_time : float, optional
        Time interval between pattern updates. Default is ``20 * dt``.

    Notes
    -----
    The controller algorithm:

    1. **Time Management**: Track simulation time and update patterns periodically
    2. **Pattern Generation**: Create wedge-shaped light patterns behind each agent
    3. **Image Creation**: Use PIL to draw patterns on a digital canvas
    4. **Interpolation Setup**: Create interpolator for smooth light evaluation
    5. **Light Evaluation**: Return light intensity at current agent positions

    The patterns are dynamic and respond to agent movement, creating a feedback
    loop where agent behavior influences the light field they experience.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        AngularFeedback:
            dt: 0.01
            line_width: 5
            angle_width: 0.785  # π/4 radians (45 degrees)
            agent_distance: 15
            update_time: 0.5

    Applications include:
    - Dynamic visual feedback systems
    - Agent-responsive light environments
    - Collective behavior with environmental coupling
    - Adaptive spatial guidance systems
    """

    def __init__(self, population, environment, config_path=None) -> None:
        """
        Initialize the angular feedback controller.

        Parameters
        ----------
        population : Population
            The population for which angular light patterns will be generated.
        environment : Environment
            The environment providing spatial boundaries.
        config_path : str, optional
            Path to the configuration file. Default is None.
        """

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
        Generate and evaluate the current angular light pattern.

        This method updates the dynamic light pattern based on current agent positions
        and returns the light intensity values experienced by each agent.

        Returns
        -------
        np.ndarray
            Light intensity values of shape (N, 1) where N is the number of agents.
            Values represent the blue channel intensity (0-1) at each agent's position.

        Notes
        -----
        The method operates in two phases:

        1. **Pattern Update** (if update_time has elapsed):
           - Calculate wedge patterns behind each agent
           - Draw lines using PIL imaging library
           - Update the spatial interpolator with new pattern
           
        2. **Light Evaluation** (every call):
           - Clip agent positions to environment boundaries
           - Evaluate light intensity at current positions
           - Return intensity values

        The wedge patterns are generated by drawing two lines from a point behind
        each agent to positions that create the specified angular width.
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
        Evaluate the angular light pattern at specified spatial positions.

        This method allows querying the current light pattern at arbitrary positions
        in the environment, useful for analysis and visualization of the dynamic
        feedback patterns.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (num_positions, 2) containing [x, y] coordinates
            where light intensity should be evaluated.

        Returns
        -------
        np.ndarray
            Light intensity values of shape (num_positions, 1) at the specified positions.
            Values represent the blue channel intensity (0-1) from the current pattern.

        Notes
        -----
        This method evaluates the most recently generated angular pattern at the
        specified positions. The pattern reflects the agent positions at the time
        of the last update (controlled by update_time parameter).

        Useful for:
        - Visualizing the dynamic light patterns
        - Analyzing spatial feedback structures
        - Creating movies or animations of pattern evolution
        """

        light_ity = self.interpolator(positions) 

        return light_ity[:,np.newaxis]