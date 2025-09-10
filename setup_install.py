#!/usr/bin/env python3
"""
Setup script for Arbor-o1 Living AI.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="arbor-o1-living-ai",
        packages=find_packages(),
        package_dir={"": "."},
    )
