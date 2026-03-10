#!/bin/bash
# Run determinism test from project root
cd "$(dirname "$0")/.."
mkdir -p determinism_test/results
./build/test_determinism
echo "Report: determinism_test/results/REPORT.md"
