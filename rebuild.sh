#!/usr/bin/env bash

# ONLY call this script from the root of the repository via ./rebuild.sh

# rebuild python wheel
python setup.py bdist_wheel

# uninstall pip wheel
pip uninstall -y MLHelper

# extract latest version from setup.py
VERSION=$(cat setup.py | grep "version=" | cut -d "\"" -f2)
WHL_PATH=$(find dist/ -type f -name "*${VERSION}*")

# install latest build
pip install ${WHL_PATH}