#!/bin/bash

# This script is used to build the cf_cpu project.
CURR_DIR=$(shell pwd)

CMAKE_HOME=$(CURR_DIR)/..
CMAKE_BUILD_DIR=$(CMAKE_HOME)/build

# Build the cf_cpu project.
all:
	@echo "Build the cf_cpu project."
	if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir $(CMAKE_BUILD_DIR); fi
	@cd $(CMAKE_BUILD_DIR) && cmake $(CMAKE_HOME) && make
	cp $(CMAKE_BUILD_DIR)/*.so $(CURR_DIR)

run:
	mpirun -n 4 python -m mpi4py main.py --config ./benchmarks/Yelp18/MF_CCL/configs/config0.yaml > out.txt

clean:
	@echo "Clean the cf_cpu project."
	@rm -rf $(CMAKE_BUILD_DIR)
	@rm -rf $(CURR_DIR)/*.so
	@rm -rf $(CURR_DIR)/out.txt