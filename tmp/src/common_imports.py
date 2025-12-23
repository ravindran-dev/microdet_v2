from __future__ import annotations

import os
import sys
import math
import time
import json
import csv
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import GroupNorm, LayerNorm

# CPU-only is fine; we just check CUDA optionally in some places
DEVICE_CPU = torch.device("cpu")

__all__ = [
    # stdlib
    "copy",
    "os",
    "sys",
    "math",
    "time",
    "json",
    "csv",
    "random",
    "logging",
    "Path",
    "Any",
    "Dict",
    "List",
    "Tuple",
    "Optional",
    # libs
    "np",
    "cv2",
    "torch",
    "nn",
    "F",
    "Dataset",
    "DataLoader",
    "_BatchNorm",
    "GroupNorm",
    "LayerNorm",
    # helpers
    "DEVICE_CPU",
]

