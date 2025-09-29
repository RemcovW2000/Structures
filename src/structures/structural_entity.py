import copy
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Literal

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

FailureMode = tuple[str, float]


def failure_analysis(func: Callable[..., list[FailureMode]]) -> Callable[..., float]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> float:
        self = args[0]
        failure_modes = func(*args, **kwargs)
        return self.set_failure_indicators(failure_modes)

    return wrapper


class StructuralEntity(ABC):
    """
    Abstract base class for structural entities.

    Has attribute failure_indicators, a dictionary of failure modes and their indicators
    showing how close the entity is to failure in that mode. Also shows max failure
    indicator of all child objects.

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

        :return: Maximum failure indicator across all failure modes and child objects.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def set_failure_indicators(self, failure_modes: list) -> float:
        """
        Sets the failure_indicators attribute based on failure modes and child objects.
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
                max_indicator = max([v for v in child.failure_indicators.values()])
                if max_indicator > child_max_indicator:
                    child_max_indicator = max_indicator

            failure_indicators["child_max_indicator"] = child_max_indicator
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
