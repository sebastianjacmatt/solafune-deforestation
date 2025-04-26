# AI-Powered Detection of Deforestation Drivers - Solafune Competition

## About the Solafune Deforestation Challenge
This project was developed for the [Solafune Deforestation Challenge](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=overview), a global machine learning competition focused on detecting and segmenting drivers of deforestation from high-resolution satellite imagery. 

Our model was designed to identify four specific classes of human or land-use activity contributing to deforestation:
- `grassland_shrubland`
- `logging`
- `mining`
- `plantation`

The goal of the challenge is to support sustainable land-use monitoring and informed decision-making through AI-powered environmental insights.


## Running and Testing the Machine Learning Pipeline

1. **Download the Dataset**  
   Download the competition dataset from [this link](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=data&tab=).

2. **Prepare the Data**  
   - Unzip the downloaded files.
   - Place all training images into the `data/train_images/` folder.
   - Place all evaluation images into the `data/evaluation_images/` folder.
   - Place the `train_annotations.json` file directly into the `data/` folder.

3. **Generate Training Masks**  
   Run the `generate_masks.py` script to create training masks from the ground truth polygons.  
   This will generate masks and save them under the `data/train_masks/` directory.

4. **Run the Pipeline**  
   Execute the `main_train.py` script located in the top-level directory to train the model.

5. **Configuration**  
   You can adjust key parameters such as the number of epochs, scheduler settings, optimizer choice, batch size, and number of workers in the `src/config.py` file.  
   **Important:** Ensure that the `TESTING` flag is set to `False` to enable full training and evaluation.

6. **Output**  
   The pipeline will output training logs to the terminal and generate a `submission.json` file.  
   This file can be uploaded to the [Solafune competition site](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=overview) to validate the model's performance on the test data.



## Applications of Research Papers to the Project
To develop a high-performing and generalizable segmentation model for detecting deforestation drivers, we drew inspiration from three key research papers. Each provided insights into augmentation techniques and robustness strategies that informed our implementation:

- **[Object-Based Augmentation Improves Quality of Remote Sensing Semantic Segmentation](https://openreview.net/forum?id=2Mf2UAAbHR)**  
  *Inspired our object-level augmentation pipeline to improve spatial realism and semantic diversity in training data.*

- **[Improving Domain Generalization with Interpolation Robustness](https://openreview.net/forum?id=Yl_4LpR_3Z)**  
  *Motivated the use of interpolation-based augmentations to increase robustness to unseen environments and domain shifts.*

- **[Automatic Data Augmentation via Invariance-Constrained Learning](https://proceedings.mlr.press/v202/hounie23a/hounie23a.pdf)**  
  *Informed our approach to learning augmentations that preserve semantic consistency while increasing variability in the data.*

These methods were adapted and integrated into our preprocessing and training pipeline to maximize generalization performance on satellite imagery.


## Project Structure

### ASCII Representation of Structure With Descriptions 
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
│   │   ├── test_preds/            # Predicted masks or classes on test set
│   │   └── val_preds/             # Predicted masks or classes on validation set
│   ├── submissions                # JSON files for leaderboard submissions
│   │   ├── 0.57/                  # Folder named after a submission score (e.g., IoU 0.57)
│   │   └── sample_answer.json     # Example submission format given to us by competition
│   └── visualizations/            # Visualizations of predictions, masks, augmentations, etc.
│       └── vis_train/             # Visualization of training images
├── src                            # Core source code for data processing, training, etc.
│   ├── preprocessing              # Scripts for preparing and analyzing the dataset
│   │   ├── data_exploration       # Scripts to explore and visualize input data
│   │   │   ├── convert_to_geojson.py       # Converts dictionaries in specified format to GeoJSON
│   │   │   ├── data_visualization.py       # Plots about input images and spectral bands
│   │   │   ├── oba_visualization.py        # Visualization of OBA-pipeline output
│   │   │   └── plot_class_distribution.py  # Plots class distribution on training set
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
│   │   └── train_utils.py                # Helper functions for training loops and metrics
│   ├── config.py                 # Global configuration for the project (paths, hyperparams)
│   ├── dataset.py                # Custom PyTorch Dataset class for training and validation
│   ├── invariance_constrained.py # Model training with invariance constraints (if used)
│   ├── model.py                  # Model architecture and forward logic
│   └── postprocess.py            # Post-processing of raw predictions (e.g., thresholding)
├── ...                           # Other project-level files (e.g., .gitignore, enviorments)
└── main_train.py                 # Entry point script to train the model
</pre>

### Where the Research Papers are Impleneted

#### Object-Based Augmentation

The Object-Based Augmentation (OBA) pipeline is primarily implemented in the `object_based_augmentation` folder under `src/utils`.

- **`oba.py`** contains the main OBA class and handles object extraction, mask alignment, and placement logic.
- **`object_augmentation.py`** performs visual transformations like rotation, flipping, and blending when objects are pasted into new backgrounds.

To integrate OBA into training, a dedicated dataset class is defined in `dataset.py`. This class handles the loading of OBA-prepared samples and ensures compatibility with the training loop. The pipeline is designed to be modular and easy to toggle via a simple boolean flag `use_oba` in the main script, which switches between standard and OBA-enabled dataloaders.


### Invariance-Constrained Learning
The Automatic data augmentation via Invariance-Constrained Learning pipeline is implemented mainly in `src/invariance_constrained.py`
The two primary functions `independent_mh_sampler.py` (1) and `primal_dual_augmentation.py` (2) corresponds with algorithms 1 and 2 from the paper. The functions are used in `src/utils/train.utils`  with flags. If flag is activated, augmentations in the dataloader will be turned off, and a separate fit-function `invariance_constrained_fit` will instead be run. This will train the model using the method from the paper.

#### Interpolation Robustness
TODO:


## Setting up the environment
This project uses conda with a list of dependencies in the environment.yml and environment_cuda.yml file

```conda env create --name solafune-deforestation-cpu --file environment.yml```
```conda env create --name solafune-deforestation-gpu --file environment_cuda.yml```

```conda activate solafune-deforestation-cpu``` or ```conda activate solafune-deforestation-gpu``` given you preffered training device.


## Configuration
All hyperparameters, paths, and model settings are stored in `src/config.py`. Edit this file to customize training or inference behavior.
