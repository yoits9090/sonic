#!/bin/bash
# sync repo files needed by the numpy-optimizer runner to a colab node.
# usage: colab/bench/sync.sh <node>
set -e
NODE=$1
COL="colab --auth adc"
$COL upload -s "$NODE" -f src/__init__.py /content/src/__init__.py
$COL upload -s "$NODE" -f src/config.py /content/src/config.py
$COL upload -s "$NODE" -f src/random_state.py /content/src/random_state.py
$COL upload -s "$NODE" -f src/impls.py /content/src/impls.py
$COL upload -s "$NODE" -f colab/bench/run_bench.py /content/bench/run_bench.py
$COL upload -s "$NODE" -f colab/bench/probe.py /content/bench/probe.py
echo "synced to $NODE"
