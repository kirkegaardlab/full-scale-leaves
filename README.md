# full-scale-leaves 
> The repository contains scripts for creating and visualizing leaf venation networks.


## Table of Contents

- [Setup](#setup)
- [License](#license)

## Setup

### 1. Install Python
The project requires Python, and has been successfully tested on v. 3.12.7. It can be downloaded from [python.org](https://www.python.org/downloads/).

### 2. Clone the repository
```sh
git clone https://github.com/kirkegaardlab/full-scale-leaves.git
cd full-scale-leaves
```

### 3. Make a virtual environment
```sh
python -m venv env
source env/bin/activate
```

### 4. Install the required packages
This project uses ```jax```, which comes in different versions depending on, e.g., hardware, OS, etc.
Please see ```https://jax.readthedocs.io/en/latest/installation.html``` for installation instructions.
The other dependencies can be installed in the usual way:

```sh
pip install -r requirements.txt
```

### 5. Download and extract the raw image data into the ```data``` directory
The data provided here consists both of the original leaf images, as well as the segmentation probabilities.
```sh
wget -O data/images_originals.zip https://zenodo.org/records/13991723/files/images_originals.zip
wget -O data/probabilities.zip https://zenodo.org/records/13991723/files/probabilities.zip
```

### 6. Extract the files
```sh
unzip -d data/ data/images_originals.zip
unzip -d data/ data/probabilities.zip
```

## Usage
In the following, we show the whole pipeline using the leaf image file ```IMG_7627.tiff``` as an example.

### Preprocessing
In order to run the model, we first need to extract the network from the segmentation probabilities,
reduce the network size by pruning, and add (possibly zero) edges to the network:
```
cd scripts
python extract.py --im=IMG_7627
python reduce.py --im=IMG_7627
python add.py --im=IMG_7627
```

### Model
To run a simulation on the preprocessed data, do:
```
python scripts/main.py --im=IMG_7627
```
The model output will be stored in ```data/model_output```.

The three most important flags to consider are:
* ```--im```, which sets the specific image data used in the script. If no flag is given, an artificial network with a regular grid is used instead.
* ```--sf```, which sets the sink fluctuation amplitude (default = 0.0).
* ```--gpu```, which allows the use of a GPU, if it is available to the user, and if the libraries and dependencies have been installed correctly.

Additionally, to run on a full network instead of running on 10 individual clusters, set:
* ```--full```, which is particularly useful for artificial networks (as they are usually much smaller than full-scale leaf networks).

For information on all flags, consult the module ```src/param_parser.py```.

### Visualization
To visualize the model output, different scripts are provided for different purposes. Note that the same flags must be used as the ones used when generating the corresponding data.
* ```scripts/analysis/comparison.py```, which compares the input and output networks.
* ```scripts/analysis/artificial.py```, which simply plots the output artificial leaf network.
* ```scripts/analysis/flows.py```, which plots the input network with the average resulting flow directions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.