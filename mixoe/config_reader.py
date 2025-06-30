import yaml
from pathlib import Path
from typing import Any, Dict, Union


class SimpleConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        if config_dict is None:
            raise ValueError("Config dictionary cannot be None. Check if your YAML file is empty or invalid.")
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)

        if self._config is None:
            raise AttributeError(f"Config is None - cannot access attribute '{name}'")

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return SimpleConfig(value)
            return value

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        if self._config is None:
            return False
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        if self._config is None:
            return {}
        result = {}
        for key, value in self._config.items():
            if isinstance(value, SimpleConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def keys(self):
        if self._config is None:
            return []
        return self._config.keys()

    def values(self):
        if self._config is None:
            return []
        for value in self._config.values():
            if isinstance(value, dict):
                yield SimpleConfig(value)
            else:
                yield value

    def items(self):
        if self._config is None:
            return []
        for key, value in self._config.items():
            if isinstance(value, dict):
                yield key, SimpleConfig(value)
            else:
                yield key, value

    def __repr__(self) -> str:
        return f"SimpleConfig({self._config})"

    def __str__(self) -> str:
        return str(self._config)


def load_config(config_path: Union[str, Path]) -> SimpleConfig:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Config file '{config_path}' is empty or contains only comments. Please ensure it has valid YAML content.")

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file '{config_path}' must contain a YAML dictionary at the root level, got {type(config_dict)}")

    return SimpleConfig(config_dict)