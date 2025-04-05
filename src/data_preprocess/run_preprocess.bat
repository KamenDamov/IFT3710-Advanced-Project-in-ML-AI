@echo off

IF NOT EXIST "./data" (
    echo The script must be run from the root of the project, where the data folder is located.
    exit /b 1
)

echo Normalization
python -m src.data_preprocess.normalization

echo Transformations application
python -m src.data_preprocess.transformations

pause