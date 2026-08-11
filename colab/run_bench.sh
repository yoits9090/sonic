#!/bin/bash
# usage: colab/run_bench.sh <node> <script> [timeout]
NODE=$1; SCRIPT=$2; TIMEOUT=${3:-600}
colab --auth adc upload -s "$NODE" -f "$SCRIPT"
colab --auth adc exec -s "$NODE" -f "$SCRIPT" --timeout "$TIMEOUT"
