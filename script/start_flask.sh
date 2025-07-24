#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
APP_DIR=${SCRIPT_DIR}/..
PYTHON_BIN=${HOME}/.pyenv/versions/3.10.13/bin/python

pushd ${APP_DIR} > /dev/null 2>&1
${PYTHON_BIN} app.py
popd > /dev/null 2>&1
