from abc import ABC, abstractmethod


class Interaction(ABC):
    """
    Abstract base class that defines the structure for interactions between two populations.

    This class serves as an interface for all interaction models in a multi-agent system.
    It requires subclasses to implement the `get_interaction` method, which computes
    the effect that `pop2` (e.g., herders) has on `pop1` (e.g., targets).

    Parameters
    ----------
    pop1 : Population
        The first population that is influenced by the interaction.
    pop2 : Population
        The second population that applies the interaction force.

    Attributes
    ----------
    pop1 : Population
        The population affected by the interaction.
    pop2 : Population
        The population exerting the interaction force.

    Notes
    -----
    - The `get_interaction` method must be implemented in all subclasses.
    - This class is designed for interactions such as **repulsion, attraction,** and **alignment**.

    Examples
    --------
    Example of a subclass implementing a specific interaction:

    .. code-block:: python

        class HarmonicRepulsion(Interaction):
            def get_interaction(self):
                # Compute repulsion forces here
                return forces
    """

    def __init__(self, pop1, pop2) -> None:
        super().__init__()
        self.pop1 = pop1  # The affected population
        self.pop2 = pop2  # The interacting population

    @abstractmethod
    def get_interaction(self):
        """
        Computes the forces that `pop2` applies on `pop1`.

        This method must be implemented by subclasses to define the specific
        interaction between the two populations.

        Returns
        -------
        np.ndarray
            A `(N1, D)` array representing the forces exerted by `pop2` on `pop1`,
            where `N1` is the number of agents in `pop1` and `D` is the state space dimension.

        Raises
        ------
        NotImplementedError
            If called directly from the base class.
        """
        pass
