from abc import ABC, abstractmethod
from swarmsim.Utils import load_config

from swarmsim.Environments import Environment


class Renderer(ABC):
    """
    Abstract base class for real-time visualization of multi-agent simulations.

    This class provides the interface for rendering environments and agent populations
    during simulation execution. Renderers can use different visualization backends
    (Matplotlib, Pygame, etc.) to provide real-time feedback on simulation state.

    Parameters
    ----------
    populations : list of Population
        List of agent populations to visualize.
    environment : Environment
        The environment instance containing spatial and visual information.
    config_path : str
        Path to the YAML configuration file containing rendering parameters.

    Attributes
    ----------
    populations : list of Population
        List of populations being rendered.
    environment : Environment
        The simulation environment.
    config : dict
        Configuration parameters for rendering settings.
    render_dt : float
        Time delay between rendering frames in seconds.
    activate : bool
        Flag to enable/disable rendering.

    Config Requirements
    -------------------
    The YAML configuration file should contain a renderer section with:

    render_dt : float, optional
        Time delay between frames in seconds. Default is ``0.05``.
    activate : bool, optional
        Whether to enable rendering. Default is ``True``.

    Additional renderer-specific parameters depend on the implementation.

    Notes
    -----
    - Subclasses must implement the abstract methods `render()` and `close()`.
    - The renderer is called at each simulation timestep if activated.
    - Common visualization elements include:
      
      * Agent positions and orientations
      * Environment boundaries and obstacles
      * Interaction forces or fields
      * Trajectories and paths
      * Performance metrics

    """

    @abstractmethod
    def __init__(self, populations: list, environment: Environment, config_path: str):
        """
        Initialize the renderer with simulation components and configuration.

        Parameters
        ----------
        populations : list of Population
            List of agent populations to visualize.
        environment : Environment
            The simulation environment instance.
        config_path : str
            Path to the YAML configuration file containing rendering parameters.

        Notes
        -----
        Subclasses should call this constructor to initialize common attributes
        and then set up their specific visualization backend (windows, graphics contexts, etc.).
        """
        config = load_config(config_path)

        self.config: dict = config.get('renderer', {})
        self.populations: list = populations
        self.environment: Environment = environment

        self.render_dt = self.config.get('render_dt', 0.05)
        self.activate = self.config.get('activate', True)

    @abstractmethod
    def render(self):
        """
        Render the current state of the simulation with suitable visualization.

        This method should update the visual representation to show the current
        positions and states of all agents, environment features, and any additional
        visualization elements (forces, trajectories, metrics, etc.).

        Implementation Requirements
        ---------------------------
        Subclasses may implement the following rendering pipeline:

        1. **Environment Rendering**: Draw boundaries, obstacles, and spatial features
        2. **Agent Visualization**: Render all populations with appropriate styling
        3. **Interaction Visualization**: Show forces, connections, or influences (optional)
        4. **Overlay Elements**: Add goal regions, trajectories, or analysis data (optional)
        5. **Frame Management**: Handle timing and display updates

        

        Notes
        -----
        - Called at each simulation timestep when rendering is active
        - Should respect the ``self.activate`` flag to enable/disable rendering
        - Performance critical method - optimize for real-time execution
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up rendering resources and shut down visualization.

        This method should properly shut down the visualization backend and release
        any allocated resources such as graphics contexts, windows, memory buffers,
        GPU resources, or file handles. It ensures clean termination of the rendering
        system and prevents resource leaks.

        Cleanup Tasks
        -------------
        **Graphics Resources**:
        
        - Close rendering windows and graphics contexts
        - Release GPU memory and buffers
        - Shut down graphics libraries (pygame, OpenGL, etc.)
        - Clear cached surfaces and textures

        

        Notes
        -----
        - Called when simulation ends or renderer is explicitly shut down
        - Critical for preventing resource leaks in long-running simulations
        """
        pass
