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
    def __init__(self, name: name_options):
        # Failure indicator by default has 'child' for the potential child object(s)
        # of the class, how can I add more failure indicators for different failure
        # modes of the object? this would be different for each class, for example the
        # member class would have a buckling failure mode, which I would like to add
        # to this dictionary

        # furthermore, how would I overwrite  the failure indicator?
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

    def set_failure_indicator(
        self, failure_mode: failure_mode_options, failure_indicator: float
    ) -> None:
        # failure indicator is initialised as an empty dictionary
        self.failure_indicators[failure_mode] = failure_indicator

    def finalize_failure_analysis(self, failure_modes: list) -> float:
        """
        Finalizes the failure analysis by setting the failure indicators for the class.

        Parameters:
        first_ply_key (str): The key to use for the first ply failure indicator.
        child_key (str): The key to use for the child failure indicator.
        """
        if failure_modes:
            for failure_mode in failure_modes:
                key = failure_mode[0]
                FI = failure_mode[1]
                self.set_failure_indicator(key, FI)

        if self.child_objects:
            child_max_key, child_max_indicator = max(
                (
                    (key, value)
                    for child in self.child_objects
                    if child
                    for key, value in child.failure_indicators.items()
                    if isinstance(value, (int, float))
                ),
                key=lambda x: x[1],
                default=(None, 0),
            )

            if child_max_key[0:6] == "child_":
                child_key = child_max_key
            else:
                child_key = "child_"
                child_key += child_max_key

            self.set_failure_indicator(child_key, child_max_indicator)

        return max(
            value
            for key, value in self.failure_indicators.items()
            if isinstance(value, (int, float))
        )

    def get_hierarchy(self, return_full: bool = False) -> dict | None:
        """
        Returns the lower hierarchy of child objects
        :return:
        """
        if return_full:
            hierarchy_dict = copy.deepcopy(self.failure_indicators)
            hierarchy_dict["object_name"] = self.name
            hierarchy_dict["children"] = []

            for child in self.child_objects:
                if child:
                    child_hierarchy_dict = child.get_hierarchy()

                    hierarchy_dict["children"].append(child_hierarchy_dict)
            return hierarchy_dict
        else:
            # only return something if the max FI > 1
            # Find the key with the maximum value
            max_key = max(self.failure_indicators, key=self.failure_indicators.get)

            # Find the maximum value
            max_value = self.failure_indicators[max_key]
            if max_value >= 1:
                hierarchy_dict = copy.deepcopy(self.failure_indicators)
                hierarchy_dict["object_name"] = self.name
                hierarchy_dict["children"] = []

                for child in self.child_objects:
                    if child:
                        child_hierarchy_dict = child.get_hierarchy()
                        if child_hierarchy_dict:
                            hierarchy_dict["children"].append(child_hierarchy_dict)
                return hierarchy_dict
            else:
                return None
