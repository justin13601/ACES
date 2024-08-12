"""This is the main module of the ACES package.

It contains the main functions and classes needed to extract cohorts.
"""
from importlib.metadata import PackageNotFoundError, version

__package_name__ = "es-aces"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:
    __version__ = "unknown"
