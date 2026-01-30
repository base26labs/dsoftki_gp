#!/bin/bash

echo "=========================================="
echo "Downloading and preparing datasets..."
echo "=========================================="

pushd data

    echo ""
    echo "Downloading MD22 datasets..."
    python get_md22.py

    echo ""
    echo "Downloading UCI datasets..."
    python get_uci.py

    echo ""
    echo "Creating N-body datasets..."
    ./create_nbody.sh

popd

echo ""
echo "=========================================="
echo "All datasets downloaded successfully!"
echo "=========================================="