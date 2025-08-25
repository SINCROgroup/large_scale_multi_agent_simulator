# 🤖 IntelliSwarms: a large scale multi-agent simulator (inSwarm)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-Alpha-orange.svg)]()

A comprehensive agent-based simulation framework designed for large-scale multi-agent systems with complex interactions, dynamic environments, and real-time control capabilities.

## Key Features

- **Multiple Populations**: Simulate heterogeneous agent populations with distinct behaviors
- **Complex Interactions**: Support for various interaction models (repulsion, attraction, shepherding)
- **Dynamic Environments**: Configurable environments with obstacles and boundaries
- **Real-time Control**: Implement controllers for real-time agent behavior modification
- **Multiple Integrators**: Various numerical integration schemes (Euler-Maruyama, etc.)
- **Visualization**: Real-time rendering with customizable display options
- **Data Logging**: Comprehensive logging system for analysis and post-processing
- **GUI Interface**: User-friendly Streamlit web interface for easy simulation setup
- **Gymnasium Integration**: RL environment support for reinforcement learning applications

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SINCROgroup/large_scale_multi_agent_simulator.git
   cd large_scale_multi_agent_simulator
   ```

2. **Install dependencies:**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using uv (recommended)
   uv pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

### Running Simulations

#### Option 1: Web GUI (Recommended for Beginners)

Launch the interactive web interface:

```bash
streamlit run streamlit_gui.py
```

Then open your browser to `http://localhost:8501` and:
1. Select a simulation type from the sidebar
2. Adjust parameters in the configuration panel
3. Click "🚀 Run Simulation" to start
4. Monitor progress and view real-time output

#### Option 2: Command Line

Run pre-configured examples directly:

```bash
# Basic multi-population simulation
python Examples/base_launcher.py

# Biological behavior simulation
python Examples/bio_launcher.py

# Shepherding simulation
python Examples/shepherding_launcher.py

# Single population simulation
python Examples/single_launcher.py

# RL environment example
python Examples/shepherding_gym_launcher.py
```

## 📖 Available Simulations

### 1. **Base Simulation** (`base_launcher.py`)
- Multi-population system with Brownian Walkers and a Fixed population 
- Harmonic repulsion interactions
- Demonstrates basic framework capabilities

### 2. **Bio Simulation** (`bio_launcher.py`)
- Biologically-inspired agent behaviors (Photosensitive microorganisms)
- Complex Agent - Environment interactions
- Environmental boundaries

### 3. **Shepherding Simulation** (`shepherding_launcher.py`)
- Shepherd-flock dynamics
- Collective behavior emergence
- Goal-directed agent movement

### 4. **Single Population** (`single_launcher.py`)
- Simple single-population dynamics
- Perfect for learning the framework
- Minimal configuration requirements

### 5. **Shepherding Gym** (`shepherding_gym_launcher.py`)
- Reinforcement Learning environment
- Gymnasium-compatible interface
- Training shepherding agents with RL algorithms

## ⚙️ Configuration

Simulations are configured using YAML files in the `Configuration/` directory. Key configuration sections include:

- **Populations**: Agent count, initial conditions, behavior parameters
- **Environment**: Boundaries, background settings
- **Interactions**: Interaction strengths, ranges, and types
- **Simulation**: Time steps, duration, integration methods
- **Rendering**: Visualization settings, colors, update rates
- **Logging**: Output formats, save frequencies, file paths

### Example Configuration Structure:

```yaml
Population_BrownianMotion:
  N: 100                    # Number of agents
  state_dim: 2             # State dimension (2D/3D)
  dt: 0.01                 # Time step
  
Environment:
  background_color: [255, 255, 255]
  render_mode: "pygame"
  
Simulation:
  T: 10.0                  # Total simulation time
  log_freq: 10             # Logging frequency
```

## 📊 Output and Analysis

Simulation results are saved to the `logs/` directory with timestamps:

- **CSV files**, **NPZ files**, **MAT files**: Data of the simulation
- **txt files**: Human readable description of the simulation

### Data Structure:
```
logs/YYYYMMDD_HHMMSS_SimulationType/
├── positions.csv           # Agent trajectories
├── states.npz             # Full state information
├── config_backup.yaml     # Configuration used
└── simulation_log.txt     # Runtime information
```

## 🔧 Advanced Usage

### Creating Custom Populations

```python
from swarmsim.Populations import BasePopulation

class CustomPopulation(BasePopulation):
    def __init__(self, config_path):
        super().__init__(config_path)
        # Custom initialization
    
    def update_state(self, dt, interactions):
        # Custom dynamics implementation
        pass
```

### Implementing Custom Interactions

```python
from swarmsim.Interactions import BaseInteraction

class CustomInteraction(BaseInteraction):
    def __init__(self, pop1, pop2, config_path):
        super().__init__(pop1, pop2, config_path)
    
    def compute_force(self):
        # Custom interaction forces
        return force_vector
```

### Real-time Control

```python
from swarmsim.Controllers import BaseController

class CustomController(BaseController):
    def __init__(self, population, config_path):
        super().__init__(population, config_path)
    
    def control_step(self, current_state):
        # Real-time control logic
        return control_input
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request


## 📚 Documentation

- **Full Documentation**: [https://sincrogroup.github.io/large_scale_multi_agent_simulator/](https://sincrogroup.github.io/large_scale_multi_agent_simulator/)


## Getting Help:

- Check the [documentation](https://sincrogroup.github.io/large_scale_multi_agent_simulator/)
- Create a new issue with detailed information about your problem
- Contact the developers

## 📄 License

This project is private and proprietary. All rights reserved.

## 👥 Authors & Maintainers

- **Stefano Covone** - s.covone@ssmeridionale.it
- **Italo Napolitano** - i.napolitano@ssmeridionale.it  
- **Davide Salzano** - davide.salzano@unina.it



