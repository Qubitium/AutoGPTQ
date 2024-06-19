#!/bin/bash

# force ruff/isort to be same version as setup.py
pip install -U ruff==0.4.9 isort==5.13.2

ruff check ../gptqmodel ../examples ../tests ../setup.py --fix
isort -l 119 -e ../
