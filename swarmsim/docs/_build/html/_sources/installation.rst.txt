Installation Guide
==================

This guide will help you install SwarmSim and its dependencies on your system.

Requirements
------------

**Python Version**
    SwarmSim requires Python 3.10 or later. We recommend using Python 3.12+ for the best performance and feature support.

**Operating Systems**
    - Windows 10/11
    - macOS 10.14+
    - Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)

Core Dependencies
-----------------

SwarmSim depends on the following core Python packages:

.. code-block:: text

    numpy >= 1.20.0
    scipy >= 1.7.0
    pandas >= 1.3.0
    matplotlib >= 3.4.0
    pyyaml >= 5.4.0
    pathlib >= 1.0.0

Optional Dependencies
---------------------

For enhanced functionality:

.. code-block:: text

    # For OpenAI Gym integration
    gymnasium >= 0.26.0

    # For advanced visualization
    plotly >= 5.0.0
    seaborn >= 0.11.0

Installation Methods
--------------------

Method 1: Install from Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the repository:**

   .. code-block:: bash

       git clone https://github.com/SINCROgroup/large_scale_multi_agent_simulator.git
       cd large_scale_multi_agent_simulator

2. **Create a virtual environment:**


   Using venv:

   .. code-block:: bash

       python -m venv swarmsim_env
       # On Windows:
       swarmsim_env\Scripts\activate
       # On macOS/Linux:
       source swarmsim_env/bin/activate

3. **Install SwarmSim:**

   .. code-block:: bash

       pip install -e .

   This installs SwarmSim in development mode, allowing you to modify the source code.

Verification
------------

Test your installation:

.. code-block:: python

    import swarmsim
    print(f"SwarmSim version: {swarmsim.__version__}")

    # Test basic functionality
    from swarmsim.Populations import BrownianMotion
    from swarmsim.Simulators import BaseSimulator

    # Create a simple simulation
    population = BrownianMotion(n=10, x_dim=2)
    simulator = BaseSimulator(populations=[population])

    # Run a few steps
    for _ in range(10):
        simulator.step()

    print("Installation successful!")

If this runs without errors, your installation is working correctly.


Next Steps
----------

After installation:

1. Read the :doc:`quickstart` guide
2. Explore the example scripts in the ``Examples/`` directory
3. Check out the API documentation for detailed usage information
4. Join our community discussions on GitHub

Support
-------

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/SINCROgroup/large_scale_multi_agent_simulator/issues>`_
2. Search for existing solutions or create a new issue
3. Include your OS, Python version, and error messages

Contributing
------------

We welcome contributions! See the development installation section above
and our contributing guidelines on GitHub.