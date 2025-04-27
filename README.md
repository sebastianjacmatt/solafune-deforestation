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
│   ├── background_images_set_separate/   # Background-only satellite image(s), a separate set for OBA 
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
|   |   |   |── ir_visualization.py         # Plots domain split image
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
│   │   ├── icl_hp_tune.py                # Hyperparameter tuning for Invariance-Constrained Learning (ICL)
│   │   ├── inference_utils.py            # Inference functions and postprocessing steps
│   │   └── train_utils.py                # Helper functions for training loops and metrics
│   ├── config.py                 # Global configuration for the project (paths, hyperparams)
│   ├── dataset.py                # Custom PyTorch Dataset class for training and validation
│   ├── invariance_constrained.py # Model training with invariance constraints (if used)
│   ├── model.py                  # Model architecture and forward logic
|   |── ir_model.py               # Model with (IR) spesific logic            
│   └── postprocess.py            # Post-processing of raw predictions (e.g., thresholding)
├── ...                           # Other project-level files (e.g., .gitignore, enviorments)
└── main_train.py                 # Entry point script to train the model
</pre>

### Description of Research Paper Implementation and Integration

#### Object-Based Augmentation (OBA)

**Method description:**  
Object-Based Augmentation (OBA) is a data augmentation technique where segmented foreground objects (e.g., plantations, grassland, mining) are extracted from satellite imagery and pasted onto new background regions.  
This increases object diversity and improves the model’s ability to generalize to unseen spatial patterns, particularly in data-scarce settings.

The method ensures semantic consistency by preventing pasted objects from overlapping existing deforestation classes. Random transformations (such as flipping and rotation) are applied to the objects during placement to further enhance variability.

**Implementation:**  
OBA is implemented in the `src/utils/object_based_augmentation/` directory:

- **`oba.py`**: Defines the `OBA` class, handling object extraction, mask alignment, placement, and overlap checks.
- **`object_augmentation.py`**: Applies visual transformations (rotation, flipping, blending) to objects before pasting.

OBA is integrated into the training pipeline via a custom dataset class in **`dataset.py`**, which mixes real and OBA-augmented samples during training.

The OBA pipeline is modular and can be activated by setting the `use_oba` flag in the main training script. This switches the dataloader to include synthetic OBA samples automatically.


#### Invariance-Constrained Learning (ICL)

**Method description:**  
Invariance-Constrained Learning (ICL) is an automatic data augmentation approach where augmentations are selected dynamically based on how well the model maintains semantic consistency after transformations.  
Instead of applying fixed augmentations, the method uses a Monte Carlo Markov Chain (MCMC) sampler to propose and accept/reject augmentations based on their effect on the model's loss, promoting invariance to transformations that the model initially struggles with.

**Implementation:**  
The ICL pipeline is mainly implemented in the `src/invariance_constrained.py` file:

- **`independent_mh_sampler.py`**: Implements the Metropolis-Hastings-based sampling of augmentations (corresponds to Algorithm 1 in the paper).
- **`primal_dual_augmentation.py`**: Defines the primal-dual optimization loop to enforce invariance constraints (corresponds to Algorithm 2 in the paper).

ICL is integrated into the training pipeline inside `src/utils/train_utils.py`:
- If the `use_icl` flag is enabled, standard augmentations in the dataloader are turned off.
- A specialized training function `invariance_constrained_fit` is used instead of the default fit function, applying the ICL method during training.


#### Interpolation Robustness (IR)

**Method description:**  
Interpolation Robustness (IR) addresses domain shift by encouraging the model to generalize between interpolated representations of different domains. In this project, spectral bands are treated as domains. The interpolation process is adapted to segmentation tasks, treating them as pixel-wise classification.

**Implementation:**  
The IR pipeline is primarily implemented in a separate model in `src/ir_model.py`:
- **`interpolation_step()`**: Computes the forward pass using interpolated features between two domain-specific images (adapted from the original paper's equations).
- **`int_loss()`**: Calculates the interpolation loss, combining classification and regularization terms.

Key components and modifications:
- **`Tψ`** (learned interpolation function) is defined and imported from `config.py`.
- The interpolation equations are adapted for the segmentation task but follow the spirit of the original Interpolation Robustness method.

Integration into the training code:
- New flags are added to `src/main_train.py` and `src/train_utils.py` to enable or disable IR during training.
- Conditional dataloading logic is introduced in `src/dataset.py` to load two domain-specific images per sample.
- Helper functions for creating domain-specific images are defined in `src/data_utils.py` (see `domain_image_split()`).

Currently, image channels are treated as separate domains. Out-of-domain channels are handled by padding—either with zeros or repetition—to maintain consistent input dimensions.

**Additional information about running Interpolation Robustness:**  
The IR pipeline is resource-intensive:
- Set `PIN_MEMORY = False` in your DataLoader settings to reduce memory pressure.
- It is also recommended to lower other memory-related parameters in `src/config.py` to avoid out-of-memory (OOM) issues during training.



## Setting up the environment
This project uses conda with a list of dependencies in the environment.yml and environment_cuda.yml file

```conda env create --name solafune-deforestation-cpu --file environment.yml```
```conda env create --name solafune-deforestation-gpu --file environment_cuda.yml```

```conda activate solafune-deforestation-cpu``` or ```conda activate solafune-deforestation-gpu``` given you preffered training device.
