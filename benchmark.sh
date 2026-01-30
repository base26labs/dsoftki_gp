#!/bin/bash

# Run benchmark scripts
pushd exp
    ./run_synthetic.sh
    ./run_md22.sh
popd
