# shadowgrouping

`shadowgrouping` is a Python package containing the measurement scheme `ShadowGrouping` of Ref. https://arxiv.org/abs/2301.03385. In addition, the used numerical benchmarks of previous state-of-the-art methods have been unified within this package.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Using the environment](#using-the-environment)
- [License](#license)

# Overview
``shadowgrouping`` 
The package utilizes a simple class structure for the measurement schemes as well the various energy estimators that come along with them.
The package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows) from this GitHub repo, see below.

# Documentation
There is no official documentation, but all classes within the package have been documented individually.
We refer to the `tutorial.ipynb` for usage of the package.

# System Requirements
## Hardware requirements
`shadowgrouping` package requires only a standard computer with enough RAM to support the in-memory operations.
Note that due to the exponentially growing Hilbert space with the molecules' qubit numbers, this required memory and the run-time of the computer also grows exponentially.
Nevertheless, the minimal working example in `tutorial.ipynb` finishes calculations within minutes on a standard computer.

## Software requirements
### OS Requirements
This package is supported for *Linux*, but other platforms should work as well. The package has been tested on the following system:
+ Linux: Ubuntu 20.04 LTS using `python3.9.18`

### Python Dependencies
`shadowgrouping` depends on a plethora of Python scientific libraries which can be found in `requirements.txt`.

# Installation Guide:

- Install `python3.9.18` from https://www.python.org/downloads/release/python-3918/ (Last checked: 29-8-23). Depending on your system, this may take a few more minutes. You may use, e.g.,
```
sudo apt update
sudo apt install python3.9
sudo apt-get install python3.9-dev python3.9-venv
```
- Pull the package and data from this repository
```
git clone https://gitlab.com/GreschAI/shadowgrouping
cd shadowgrouping
python3.9 -m venv .shadowgrouping_env
source .shadowgrouping_env/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```
Again, this installation process depends on your system and internet speed, but should be done after a few minutes.

# Using the package:
- To run demo notebooks:
  - `jupyter notebook`
  - Then copy the url it generates, it looks something like this: `http://localhost:8889/?token=dde30ccc772afed3012e7c3be67a537cc1ea9036c22357c8`
  - Open it in your browser
  - Then open `tutorial.ipynb` which includes the minimal working example. Running all executable code in the notebooks sequentially should not take more than a few minutes on a standard laptop. If other molecules are selected, however, this run time can easily turn into a few hours though.

# License

This project is covered under the **Apache 2.0 License**.
