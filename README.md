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

- Implement (https://github.com/motokimura/solafune_deforestation_baseline) in our structure
- Use different a pretrained model that works better with sattelite data
```py
self.model = smp.create_model(
            arch="unet",
            encoder_name="tf_efficientnetv2_s",  # <-- this pre-trained model is trained on a dataset of [dogs](https://www.image-net.org/)💀
            encoder_weights="imagenet",  # always starts from imagenet pre-trained weight
            in_channels=12,
            classes=4,
        )
```
Try and produce some JSON data, and send in solution


for later -->
- UNET-convocational neural network scheme for segmentation
- Transfer learning of a previosuly trained segmentation model (preferably one working with the same channels as the ones in solafune data)
**Models Selection**
- Ensamble different models for different channels
- implement performance metric from solafune competition
