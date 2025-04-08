#!/bin/bash

BATCH_SIZE=100  # Change this to your desired batch size
N_PARTITION=10
FEATURES_FILE="/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/src/data_preprocess/modalities/features_list.pkl"

# Delete the file if it exists
if [ -f "$FEATURES_FILE" ]; then
    echo "Deleting existing $FEATURES_FILE"
    rm "$FEATURES_FILE"
fi

for (( PARTITION=0; PARTITION<N_PARTITION; PARTITION++ ))
do
    echo "Running partition $PARTITION / $((N_PARTITION - 1)) with batch size $BATCH_SIZE"
    if [ "$PARTITION" -eq "$((N_PARTITION - 1))" ]; then
        python find_modalities.py --partition "$PARTITION" --batch_size "$BATCH_SIZE" --do_get_modalities True
    else
        python find_modalities.py --partition "$PARTITION" --batch_size "$BATCH_SIZE"
    fi
done