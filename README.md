# IFT3710-Advanced-Project-in-ML-AI
The aim of the project will be to combine GAN architectures to generate biological cell segmentation.
We will also compare this new architecture with state-of-the-art models, as well as with more rudimentary models.

# SEE THIS GOOGLE DRIVE FOR TRAINED MODELS, PROCESSED DATASETS AND OTHER ARTIFACTS:
https://drive.google.com/drive/folders/11TrqoDSULiYNIRG8w6ky_wyWjcGsxA3y

## Dataset preparation and pre-processing

Make sure to download the datasets and pre-trained models from their respective websites:

MEDIAR pre-trained models:
# https://drive.google.com/drive/folders/1RgMxHIT7WsKNjir3wXSl7BrzlpS05S18

NeurIPS Competition dataset:
# https://zenodo.org/records/10719375

CellPose dataset:
# https://www.cellpose.org/dataset

LIVECell dataset:
# https://github.com/sartorius-research/LIVECell
# http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

OmniPose dataset:
# https://osf.io/xmury/files/osfstorage

ScienceBowl 2018 dataset:
# https://www.kaggle.com/competitions/data-science-bowl-2018/data (for data zip)
# https://bbbc.broadinstitute.org/BBBC038/ (for stage2 csv)

Then organize the downloaded files into the following folders:

# /models/mediar/pretrained
#               └── phase1.pth    
# /data
# ├── /raw
# |   ├── /cellpose       
# |   |   ├── test.zip              
# |   |   ├── train.zip             
# |   |   └── train_cyto2.zip       
# |   ├── /livecell       
# |   |   ├── livecell_coco_test.json           
# |   |   ├── livecell_coco_train.json         
# |   |   ├── livecell_coco_val.json           
# |   |   └── images.zip                        
# |   ├── /neurips        
# |   |   ├── Testing.zip                   
# |   |   ├── Training-labeled.zip          
# |   |   ├── train-unlabeled-part1.zip     
# |   |   ├── train-unlabeled-part2.zip     
# |   |   ├── Tuning.zip                    
# |   |   └── ReadMe.md                 
# |   ├── /omnipose       
# |   |   └── datasets.zip
# |   |       ├── /bact_fluor   
# |   |       ├── /bact_phase  
# |   |       ├── /worm
# |   |       └── /worm_high_res        
# |   └── /sciencebowl    
# |       ├── stage2_solution_final.csv            
# |       └── data-science-bowl-2018.zip             
# |           └── *various zip files*        
# ├── dataset.images.csv  (generated)
# ├── dataset.labels.csv  (generated)
# └── features.pkl        (generated)

IMPORTANT:
The collected zip files from the web generally contain a single folder of the same name.
That is expected, but there are some exceptions (like omnipose). Just keep them in the same format as they were found.
Our scripts do not depend on the zip file names, but they do depend on the deeper folder structure.

From the root directory of the Git repository, run the following command (on Windows):
> ./src/data_preprocess/run_preprocess.bat

