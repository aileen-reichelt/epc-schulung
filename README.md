# EPC Schulung
A project describing the setup for the EPC machine learning training.

## Setup

Setup Python for ML from scratch.

### Installing Anaconda
Anaconda is a Python distribution, package and environment manager, and data science toolkit. Follow the [installation guidelines](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) for your OS.

### Cloning
Clone this repository with `git clone https://github.com/aileen-reichelt/epc-schulung.git` or, using ssh, `git clone git@github.com:aileen-reichelt/epc-schulung.git` to a location of your choice. (Or click the green button "Code" and download a ZIP file of the repository's contents.) 

### Python and Python packages
Python and its desired packages are included with Anaconda (or rather, Anaconda can install them easily while resolving for compatibility issues). Anaconda manages virtual environments, which can be exported and imported. Create an Anconda environment with all dependencies (including Python) necessary for this project from the `.yml` file provided in this repository. To do so:

**On Windows** start an "Anaconda Prompt" window (after installing Anaconda), **on Linux**, open a terminal window for the following command.

```bash
cd epc-schulung  # go to project repo
conda env create -f schulung_crossplattform.yml  # confirm any installation prompts
conda activate schulung  # (to deactivate, use conda deactivate)
conda env list  # verify correct installation
```
