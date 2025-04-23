# solafune-deforestation

## About the Solafune Deforestation Challenge

This project was developed for the [Solafune Deforestation Challenge](https://solafune.com), a global machine learning competition focused on detecting and segmenting drivers of deforestation from high-resolution satellite imagery. 

Our model was designed to identify four specific classes of human or land-use activity contributing to deforestation:
- `grassland_shrubland`
- `logging`
- `mining`
- `plantation`

The goal of the challenge is to support sustainable land-use monitoring and informed decision-making through AI-powered environmental insights.


## Applications of research papers to the project

To create the best possible model for our project, we implemented three different research papers

## setting up the environment

This project uses conda with a list of dependencies in the environment.yml file

```conda env create --name solafune-deforestation --file environment.yml```

```conda activate solafune-deforestation```

## Adding a dependency

Here is how you can add a new dependency or environment variable:

add you dependency or environment variable in the environment.yml file like so:

```yml
name: solafune-deforestation
channels:
  - defaults
variables:
  ./data # added a new environment variable
dependencies:
  - python=3.11
  - numpy
  - scipy
  - scikit-learn
  - pandas
  - jupyter
  - ipykernel
  - pickleshare
  - matplotlib
  - pillow # here i added pillow
```

Then you update the environment, make sure you are in the root directory and run the command:

```conda env update -f environment.yml --prune && conda deactivate && conda activate solafune-deforestation```


## Project Structure
<pre>
.
├── data                           # Contains all input data for training and evaluation
│   ├── background_images/         # Background-only satellite image(s), a separate set for OBA 
│   ├── evaluation_images/         # Unlabeled images used for model evaluation or testing
│   ├── train_images/              # Original satellite images for training
│   ├── train_masks/               # Ground truth segmentation masks for training images
│   └── train_annotations.json     # Annotations for training set
├── models                         # Stores trained models and checkpoints
│   └── checkpoints/               # Saved weights from training epochs or best models
├── outputs                        # All model output files (e.g., predictions, visualizations)
│   ├── predictions                # Raw model predictions on validation/test data
│   │   └── val_preds/             # Predicted masks or classes on validation set
│   ├── submissions                # JSON files for leaderboard submissions
│   │   ├── 0.57/                  # Folder named after a submission score (e.g., IoU 0.57)
│   │   └── sample_answer.json     # Example submission format given to us by competition
│   └── visualizations/            # Visualizations of predictions, masks, augmentations, etc.
│       └── vis_train/             # Visualization of training images
├── src                            # Core source code for data processing, training, etc.
│   ├── preprocessing              # Scripts for preparing and analyzing the dataset
│   │   ├── data_exploration       # Scripts to explore and visualize input data
│   │   │   ├── data_visualization.py     # Plots about input images and spectral bands
│   │   │   └── oba_visualization.py      # Visualization of OBA-pipeline output
│   │   └── mask_generation        # Tools to create and manipulate segmentation masks
│   │       ├── generate_masks.py         # Pipeline to generate masks from annotations
│   │       ├── get_masks.py              # Helper functions to fetch or format masks
│   │       └── visualize_masks.py        # Visual debugging of generated masks
│   ├── utils                      # Utility scripts and modules
│   │   ├── object_based_augmentation     # OBA module for cut-and-paste data augmentation
│   │   │   ├── oba.py                     # Main class for handling OBA logic
│   │   │   └── object_augmentation.py     # Augmentations applied to pasted objects
│   │   ├── data_utils.py                 # General-purpose data loading and manipulation
│   │   ├── global_paths.py               # Centralized paths used across modules
│   │   ├── inference_utils.py            # Inference functions and postprocessing steps
│   │   ├── oba_augmentation.py           # Pipeline for applying full OBA logic
│   │   └── train_utils.py                # Helper functions for training loops and metrics
│   ├── augmentation.py           # Augmentation strategies applied to training data
│   ├── config.py                 # Global configuration for the project (paths, hyperparams)
│   ├── dataset.py                # Custom PyTorch Dataset class for training and validation
│   ├── invariance_constrained.py # Model training with invariance constraints (if used)
│   ├── model.py                  # Model architecture and forward logic
│   └── postprocess.py            # Post-processing of raw predictions (e.g., thresholding)
├── ...                           # Other project-level files (e.g., .gitignore, enviorments)
└── main_train.py                 # Entry point script to train the model
</pre>



## TODO

CREATE DOCSTRINGS/COMMENTS FOR EVERY FUNCTION IN ENTIRE CODEBASE
ADD EXPLANATION ABOUT THE COMPETITION AND BACKGROUND INFORMATION ABOUT THE PROJECTTO README
