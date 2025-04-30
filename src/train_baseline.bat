@echo off

IF NOT EXIST "./data" (
    echo The script must be run from the root of the project, where the data folder is located.
    exit /b 1
)

echo Move normalized image files
python -m src.train_baseline

cd "./src/NeurIPS-CellSeg"
python -m baseline.model_training_3class
