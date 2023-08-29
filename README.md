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

- Install `python3.9.18` from https://www.python.org/downloads/release/python-3918/ (Last checked: 29-8-23)
```
git clone https://gitlab.com/GreschAI/shadowgrouping
cd shadowgrouping
python3.9 -m venv .shadowgrouping_env
source .shadowgrouping_env/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

# Using the package:
- To run demo notebooks:
  - `jupyter notebook`
  - Then copy the url it generates, it looks something like this: `http://(0de284ecf0cd or 127.0.0.1):8888/?token=e5a2541812d85e20026b1d04983dc8380055f2d16c28a6ad`
  - Open it in your browser
  - Then open `tutorial.ipynb` which includes the minimal working example

# License

This project is covered under the **Apache 2.0 License**.
