import copy
from abc import ABC, abstractmethod
from typing import Literal

failure_mode_options = Literal[
    "fiber_failure",
    "inter_fiber_failure",
    "first_ply_failure",
    "buckling",
    "wrinkling",
    "child",
]

name_options = Literal[
    "lamina",
    "laminate",
    "sandwich",
    "member",
    "airfoil",
    "wing",
]

failure_state_options = Literal[True, False]


class StructuralEntity(ABC):
    """
    Abstract base class for structural entities.

    Has attribute failure_indicators, a nested dictionary that stores the entity's own
     failure indicators and those of its children.

    Enforces implementation of:
    - failure_analysis method
    """

    def __init__(self, name: name_options):
        self.failure_indicators = {}
        self.name = name

    @property
    def child_objects(self) -> list["StructuralEntity"]:
        return []

    @abstractmethod
    def failure_analysis(self) -> float:
        """
        Perform failure analysis on the structural entity, call method on all children.

        Is an augmentor class -> influences the state of the class

        Sets the correct failure indicator for the right failure mode(s)

        :return: Maximum failure indicator across all failure modes and child objects.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def set_failure_indicators(self, failure_modes: list) -> float:
        """
        Finalizes the failure analysis by setting the failure indicators for the class.

        Parameters:
        first_ply_key (str): The key to use for the first ply failure indicator.
        child_key (str): The key to use for the child failure indicator.
        """
        failure_indicators = {}
        if failure_modes:
            for failure_mode in failure_modes:
                key = failure_mode[0]
                fi = failure_mode[1]
                failure_indicators[key] = fi

        if self.child_objects:
            child_max_indicator = 0
            for child in self.child_objects:
                for key, value in child.failure_indicators.items():
                    if value > child_max_indicator:
                        child_max_indicator = value
                        child_max_key = key

            if child_max_key.startswith("child_"):
                child_key = child_max_key
            else:
                child_key = "child_"
                child_key += child_max_key

            failure_indicators[child_key] = child_max_indicator
        self.failure_indicators = failure_indicators
        return max(value for value in self.failure_indicators.values())

    def get_hierarchy(self) -> dict | None:
        """Returns the lower hierarchy of child objects."""
        hierarchy_dict = copy.deepcopy(self.failure_indicators)
        hierarchy_dict["object_name"] = self.name
        hierarchy_dict["children"] = []

        for child in self.child_objects:
            child_hierarchy_dict = child.get_hierarchy()

            hierarchy_dict["children"].append(child_hierarchy_dict)
        return hierarchy_dict
