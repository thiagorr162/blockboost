# BlockBoost: Scalable and Efficient Blocking through Boosting

This repository contains the code for [BlockBoost: Scalable and Efficient Blocking through Boosting](https://proceedings.mlr.press/v238/ramos24a.html), published in AISTATS 2024.

## Package setup

- All scripts should be run from the git repository root directory;

- The root of the project must be in the `PYTHONPATH` variable, one way to do this is by running or adding to the `.profile` script the following line:
  ```sh
  export PYTHONPATH=.
  ```

- Python packages were managed by [Poetry](https://python-poetry.org/docs/), which can be installed via `pipx`. To activate and install the environment, run from the root directory:
  ```sh
    poetry shell
    poetry install
  ```

## Data setup

### Downloading

To download a datasets used in our paper, you can simply run
```
   python src/data/download/download_{dataset_name}.py
```
### Processing
By running the provided code, you can easily process the datasets used in our paper. The code performs various tasks, including performing a textual vectorization on the dataset, generating record and entity ID's for each entry, and creating train, validation, and test splits.
```
   python src/data/download/process_{dataset_name}.py
```
### Vectorization
In our paper, we describe multiple vectorization approaches that you can choose from. To execute a particular vectorization, simply use the following command:
```
   python src/data/download/vectorize_{vectorization_name}.py -db {dataset_name} [other specific vectorization arguments]
```
Within each code file, you will find detailed descriptions of the specific vectorization arguments, providing appropriate guidance for their usage.

### .sh scripts

Alternatively, you can simplify the process of processing and vectorizing multiple datasets by using the .sh scripts named vectorize_all_{vectorization_name}.sh and process_all.sh . These scripts allow you to execute vectorization on all datasets with a single command.

## Running models

### Blockboost

#### Compilation

Just run `make` at the root of the project.

Note that `openmp` and `boost` libraries are required to compile the binaries. On Debian/Ubuntu, those can be installed with

```sh
sudo apt install -y libboost-all-dev openmpi-common
```

#### Execution

```sh
./src/models/blockboost/generate_experiments-fp32.sh
```

### DeepBlocker

DeepBlocker experiments were conducted using the official repository available at https://github.com/qcri/DeepBlocker.

### Other benchmark models
To use other benchmark models, you can simply run the following command:
```
python src/models/{benchmark_model_name}/predict.py -db {vectorized_database_name} [other specific model arguments]
```
Within each model file, you will find detailed descriptions of the specific arguments, providing appropriate guidance for their usage.


## Evaluating

### Evaluating experiments

Run from the root directory,

```sh
./src/eval/eval.sh <model_name>
```

### Creating tables containing the results

To generate the .csv tables that contain the results with each metric for each dataset and benchmark model used, you can simply execute the following command:
```
python src/eval/create_eval_tables.py
```
