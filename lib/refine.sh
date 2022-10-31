#!/bin/bash
echo "Refining $1 with default parameters"
python3 lib/refine-task.py $1 10 EE_reduced
