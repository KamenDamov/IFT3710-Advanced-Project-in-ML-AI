@echo off

IF NOT EXIST "./data" (
    echo The script must be run from the root of the project, where the data folder is located.
    exit /b 1
)

echo Unpack raw training set files
python -m src.datasets.datasets

echo Clean up format of image and mask files
python -m src.data_preprocess.unify.unify

echo Normalization
python -m src.data_preprocess.normalization

echo Transformations application
python -m src.data_preprocess.transformations

pause