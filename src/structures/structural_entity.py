import copy
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Generic, Literal, Optional, TypeVar

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

T = TypeVar("T")


class cached_property(Generic[T]):
    """Descriptor that caches a computed value and optionally allows a setter.

    Usage:
        class C:
            @cached_property
            def x(self) -> int:
                return compute()

            @x.setter
            def x(self, value: int) -> None:
                # optionally handle set
                self._raw = value

    When the property is read the first time, the getter runs and the result is
    stored on the instance under `_cached_<name>`. When set, if a setter was
    provided it will be called; its return value (if not None) will be stored in
    the cache, otherwise the provided value will be stored.
    """

    def __init__(self, func: Optional[Callable[..., T]] = None, name: Optional[str] = None) -> None:
        # allow using @cached_property without calling it (func provided)
        self.fget: Optional[Callable[..., T]] = func
        self.fset: Optional[Callable[[Any, T], Optional[T]]] = None
        self.attr_name: Optional[str] = name or (func.__name__ if func is not None else None)
        self.cache_name: Optional[str] = f"_cached_{self.attr_name}" if self.attr_name else None
        self.__doc__ = getattr(func, "__doc__", None) if func is not None else None

    def __set_name__(self, owner: type, name: str) -> None:
        if self.attr_name != name:
            self.attr_name = name
            self.cache_name = f"_cached_{name}"

    def __get__(self, instance: Optional[object], owner: type) -> Any:
        if instance is None:
            return self
        if self.cache_name is None:
            raise AttributeError("cached_property not properly initialized")
        if self.cache_name in instance.__dict__:
            return instance.__dict__[self.cache_name]
        if self.fget is None:
            raise AttributeError(f"unreadable attribute '{self.attr_name}'")
        value = self.fget(instance)
        instance.__dict__[self.cache_name] = value
        return value

    def setter(self, func: Callable[[Any, T], Optional[T]]) -> "cached_property[T]":
        """Decorator to register a setter function for the cached property."""
        self.fset = func
        return self

    def __set__(self, instance: object, value: T) -> None:
        if self.fset is None:
            raise AttributeError(f"can't set attribute '{self.attr_name}'")
        # call user setter; if it returns a value, cache that, otherwise cache the provided value
        ret = self.fset(instance, value)
        if self.cache_name is None:
            raise AttributeError("cached_property not properly initialized")
        instance.__dict__[self.cache_name] = value if ret is None else ret

    def __delete__(self, instance: object) -> None:
        if self.cache_name is None:
            return
        instance.__dict__.pop(self.cache_name, None)


def failure_analysis(func: Callable[..., list[FailureMode]]) -> Callable[..., float]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> float:
        self = args[0]
        failure_modes = func(*args, **kwargs)
        return self.set_failure_indicators(failure_modes)

    # mark the wrapper so subclasses are required to use this decorator
    wrapper._is_failure_analysis = True  # type: ignore[attr-defined]
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

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if "failure_analysis" not in cls.__dict__:
            raise TypeError(
                "Subclasses must implement `failure_analysis` and decorate it with `@failure_analysis_decorator`"
            )

        impl = cls.__dict__["failure_analysis"]
        # unwrap descriptors
        if isinstance(impl, (classmethod, staticmethod)):
            impl = impl.__func__

        if not getattr(impl, "_is_failure_analysis", False):
            raise TypeError(
                "`failure_analysis` must be decorated with `@failure_analysis_decorator`"
            )

    def __init__(self, name: name_options):
        self.failure_indicators: dict[str, float] = {}
        self.name: str = name

    @cached_property
    def loads(self) -> Optional[Any]:
        return self.loads_from_strains()

    @abstractmethod
    def loads_from_strains(self) -> Any:
        raise NotImplementedError("Subclasses must implement `loads_from_strains`")

    @loads.setter
    def loads(self, value: Any) -> None:
        """Set loads, invalidate strains and cached failure indicator."""
        self.loads = value
        self.strains = None
        self.__dict__.pop("_cached_fi", None)

    @cached_property
    def strains(self) -> Optional[Any]:
        return self.strains_from_loads()

    @abstractmethod
    def strains_from_loads(self) -> Any:
        raise NotImplementedError("Subclasses must implement `strains_from_loads`")

    @strains.setter
    def strains(self, value: Any) -> None:
        """Set strains, invalidate loads and cached failure indicator."""
        self.strains = value
        self.loads = None
        self.__dict__.pop("_cached_fi", None)

    def invalidate_failure_state(self) -> None:
        """Invalidate cached properties."""
        self.__dict__.pop("_cached_fi", None)
        self.failure_indicators = {}

    @property
    def child_objects(self) -> list["StructuralEntity"]:
        return []

    @cached_property
    def fi(self) -> float:
        r"""
        Read\-only failure indicator. Runs `failure_analysis()` to update
        `failure_indicators` and returns the current maximum indicator.
        """
        return self.failure_analysis()

    @abstractmethod
    def failure_analysis(self) -> float:
        """
        Perform failure analysis on the structural entity, call method on all children.

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


if __name__ == "__main__":
    # basic usage example demonstrating loads/strains interdependence and cached fi
    class DemoEntity(StructuralEntity):
        def __init__(self, name: name_options):
            super().__init__(name)

        @failure_analysis
        def failure_analysis(self) -> list[FailureMode]:
            # very small illustrative heuristic: strains dominate, otherwise use loads
            if self.strains is not None:
                # expect a mapping with key 'eps'
                eps = self.strains.get("eps", 0) if isinstance(self.strains, dict) else 0
                fi = float(abs(eps) * 10.0)
            elif self.loads is not None:
                force = self.loads.get("force", 0) if isinstance(self.loads, dict) else 0
                fi = float(abs(force) / 100.0)
            else:
                fi = 0.0
            return [("fiber_failure", fi)]

    ent = DemoEntity("lamina")
    print("initial loads:", ent.loads, "strains:", ent.strains)

    ent.loads = {"force": 200.0}
    print("after setting loads -> loads:", ent.loads, "strains:", ent.strains)

    print("computed fi:", ent.fi)

    # setting strains invalidates loads and the cached fi
    ent.strains = {"eps": 0.05}
    print("after setting strains -> loads:", ent.loads, "strains:", ent.strains)

    print("computed fi after strains set:", ent.fi)

    # demonstrate explicit cache deletion
    del ent.fi
    print("deleted cached fi, next access recomputes:", ent.fi)
