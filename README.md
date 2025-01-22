# solafune-deforestation

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

## DONE

## TODO

- download images
- setup enviornmet & gitnore
**Preprocessing**
- polygon to pixel pixel to polygon
**Model scheme**
- UNET-convocational neural network scheme for segmentation
- Transfer learning of a previosuly trained segmentation model (preferably one working with the same channels as the ones in solafune data)
**Models Selection**
- Ensamble different models for different channels
- implement performance metric from solafune competition
