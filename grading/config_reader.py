import yaml
import os
from typing import Dict, Any


class SimpleConfig:
    """Simple configuration class that mimics basic Hydra functionality"""

def __init__(self, config_dict:Dict[str, Any]):
        self._config= config_dict
self._update_config_dict(self._config)

def _update_config_dict(self, d:Dict[str, Any], parent_key:str=''):
        """Recursively convert nested dictionaries to SimpleConfig objects"""
for key, valueind.items():
            if isinstance(value, dict):
                d[key]= SimpleConfig(value)

def __getattr__(self, name:str):
        if name.startswith('_'):
            return super().__getattribute__(name)
if namein self._config:
            return self._config[name]
raiseAttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def __setattr__(self, name:str, value:Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
else:
            if hasattr(self,'_config'):
                self._config[name]= value
else:
                super().__setattr__(name, value)

def __getitem__(self, key:str):
        return self._config[key]

def __setitem__(self, key:str, value:Any):
        self._config[key]= value

def __contains__(self, key:str):
        return keyin self._config

def get(self, key:str, default:Any= None):
        return self._config.get(key, default)

def to_dict(self)->Dict[str, Any]:
        """Convert config back to dictionary, handling nested SimpleConfig objects"""
result={}
for key, valuein self._config.items():
            if isinstance(value, SimpleConfig):
                result[key]= value.to_dict()
else:
                result[key]= value
return result


def load_config(config_path:str="config/base.yaml")->SimpleConfig:
    """Load YAML configuration file and return SimpleConfig object"""
if notos.path.exists(config_path):
        raiseFileNotFoundError(f"Config file not found: {config_path}")

with open(config_path,'r')asfile:
        config_dict= yaml.safe_load(file)

return SimpleConfig(config_dict)


def save_config(config:SimpleConfig, output_path:str)->None:
    """Save configuration to YAML file"""
os.makedirs(os.path.dirname(output_path), exist_ok= True)

with open(output_path,'w')asfile:
        yaml.dump(config.to_dict(), file, default_flow_style= False, indent=2)


def config_to_yaml_string(config:SimpleConfig)->str:
    """Convert config to YAML string for printing"""
return yaml.dump(config.to_dict(), default_flow_style= False, indent=2)