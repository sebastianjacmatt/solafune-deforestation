# solafune-deforestation

## setting up the environment

This project uses conda with a list of dependencies in the environment.yml file

```conda env create --name solafune-deforestation --file environment.yml```

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
