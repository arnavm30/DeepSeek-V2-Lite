## Installation

1. Install [Mamba](https://github.com/mamba-org/mamba):
    ```bash
    conda install mamba -n base -c conda-forge
    ```

2. Create the conda environment from the configuration file using Mamba:
    ```bash
    mamba env create -f environment.yml
    ```

3. Activate the conda environment:
    ```bash
    conda activate deepseek-lite
    ```

4. Run a test training step:
    ```bash
    python test_train_step.py
    ```
