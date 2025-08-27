Controllers Package
===================

The Controllers package provides control algorithms and strategies for guiding agent behavior
in multi-agent simulations. Controllers implement various coordination, shepherding, and 
guidance strategies that can be applied to populations of agents.

.. automodule:: swarmsim.Controllers
   :members:
   :undoc-members:
   :show-inheritance:

Package Overview
----------------

Controllers in SwarmSim enable sophisticated multi-agent coordination through:

* **Shepherding Algorithms**: Guide groups of agents toward specific goals
* **Formation Control**: Maintain desired spatial arrangements  
* **Coordination Strategies**: Implement collective decision-making
* **Adaptive Control**: Respond dynamically to changing conditions
* **Hierarchical Control**: Multi-level control architectures

All controllers inherit from the base controller interface, ensuring consistent integration
with the simulation framework.

Core Modules
------------

Base Controller Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

The base controller defines the standard interface that all controllers must implement.

.. automodule:: swarmsim.Controllers.base_controller
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Standardized Interface**: Consistent API for all controller implementations
- **Flexible Integration**: Easy integration with different agent populations
- **State Management**: Handle controller state and memory
- **Performance Monitoring**: Built-in performance tracking capabilities

Shepherding Controller
~~~~~~~~~~~~~~~~~~~~~~

Advanced shepherding algorithms for guiding agent groups toward target locations.

.. automodule:: swarmsim.Controllers.shepherding_lama_controller
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Multi-agent Shepherding**: Coordinate multiple shepherd agents
- **Adaptive Strategies**: Dynamic adjustment based on group behavior
- **Obstacle Avoidance**: Navigate around environmental obstacles
- **Performance Optimization**: Efficient algorithms for large-scale groups

Spatial Input Processing
~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for processing spatial information and environmental inputs.

.. automodule:: swarmsim.Controllers.spatial_inputs
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

- **Spatial Awareness**: Process environmental and agent spatial information
- **Sensor Integration**: Handle various sensor inputs and data fusion
- **Coordinate Transformations**: Convert between different spatial representations
- **Real-time Processing**: Efficient real-time spatial data processing

Usage Examples
--------------

Basic Controller Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Controllers import BaseController
    from swarmsim.Populations import BrownianMotion
    
    # Create population and controller
    population = BrownianMotion(n=100, x_dim=2)
    controller = BaseController()
    
    # Apply control in simulation loop
    for step in range(1000):
        control_input = controller.compute_control(population)
        population.apply_control(control_input)
        population.step()

Shepherding Control
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Controllers import ShepherdingLamaController
    from swarmsim.Populations import BrownianMotion
    import numpy as np
    
    # Setup target and populations
    target_location = np.array([10.0, 10.0])
    sheep_population = BrownianMotion(n=50, x_dim=2)
    shepherd_population = BrownianMotion(n=3, x_dim=2)
    
    # Initialize shepherding controller
    controller = ShepherdingLamaController(
        target_position=target_location,
        shepherding_gain=1.0,
        formation_gain=0.5
    )
    
    # Simulation with shepherding
    for step in range(2000):
        # Compute shepherding actions
        shepherd_actions = controller.compute_shepherding_action(
            sheep_population.x, 
            shepherd_population.x
        )
        
        # Apply control
        shepherd_population.apply_control(shepherd_actions)
        
        # Update populations
        sheep_population.step()
        shepherd_population.step()

Advanced Control Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from swarmsim.Controllers import BaseController
    from swarmsim.Controllers.spatial_inputs import SpatialInputProcessor
    import numpy as np
    
    class CustomFormationController(BaseController):
        def __init__(self, formation_shape='circle', formation_radius=5.0):
            super().__init__()
            self.formation_shape = formation_shape
            self.formation_radius = formation_radius
            self.spatial_processor = SpatialInputProcessor()
        
        def compute_control(self, population):
            # Get current positions
            positions = population.x
            n_agents = len(positions)
            
            # Define target formation
            if self.formation_shape == 'circle':
                angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
                target_positions = self.formation_radius * np.column_stack([
                    np.cos(angles), np.sin(angles)
                ])
            
            # Compute formation control forces
            position_errors = target_positions - positions
            control_forces = 0.5 * position_errors  # Proportional control
            
            return control_forces
    
    # Use custom controller
    controller = CustomFormationController(formation_shape='circle', formation_radius=8.0)

Control Architecture
--------------------

The Controllers package supports hierarchical control architectures:

.. code-block:: python

    class HierarchicalController:
        def __init__(self):
            self.high_level_controller = StrategicPlanner()
            self.low_level_controllers = [
                LocalController() for _ in range(n_groups)
            ]
        
        def compute_control(self, populations):
            # High-level strategic decisions
            strategies = self.high_level_controller.plan(populations)
            
            # Low-level tactical execution
            controls = []
            for i, (population, strategy) in enumerate(zip(populations, strategies)):
                local_control = self.low_level_controllers[i].execute(
                    population, strategy
                )
                controls.append(local_control)
            
            return controls

Performance Considerations
-------------------------

- **Computational Efficiency**: Controllers are optimized for real-time performance
- **Scalability**: Algorithms designed to handle large populations efficiently
- **Memory Usage**: Minimal memory footprint for embedded applications
- **Parallel Processing**: Support for parallel control computation

Best Practices
--------------

1. **Controller Selection**: Choose controllers appropriate for your application
2. **Parameter Tuning**: Adjust control gains for stable, responsive behavior
3. **Performance Monitoring**: Monitor controller performance and stability
4. **Modular Design**: Combine multiple controllers for complex behaviors
5. **Testing**: Validate controller behavior across different scenarios
