#!/bin/bash

module load python/3.9.2

# Run dask_preprocess.py
python dask_preprocess.py

# Check if the previous command succeeded
if [ $? -eq 0 ]
then
    # Run spark_preprocess.py if the previous command succeeded
    python spark_preprocess.py
fi
