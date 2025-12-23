# src/models/registry.py
from typing import Type, Dict

MODEL_REGISTRY: Dict[str, Type] = {}


def register_module(cls: Type):
    
    name = cls.__name__
    if name in MODEL_REGISTRY:
        raise KeyError(f"Module '{name}' already registered.")
    MODEL_REGISTRY[name] = cls
    return cls


def get_module(name: str):
   
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Module '{name}' not found in registry.")
    return MODEL_REGISTRY[name]


def list_registered():
  
    return list(MODEL_REGISTRY.keys())
