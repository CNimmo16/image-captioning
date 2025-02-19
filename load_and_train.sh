#!/bin/bash

pdm run load-data
FULLRUN=1 pdm run train
