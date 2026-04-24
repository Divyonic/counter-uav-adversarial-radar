"""Pytest configuration: put the project root on sys.path."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "adversarial"))
