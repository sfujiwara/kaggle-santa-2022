# Kaggle Santa 2022

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org)

The 37th place solution of Kaggle competition [Santa 2022 - The Christmas Card Conundrum](https://www.kaggle.com/competitions/santa-2022).

## Solution

### Summary

As mentioned in the discussion [here](https://www.kaggle.com/competitions/santa-2022/discussion/376306), we could generate arm configurations from positions in each orthants.
Thus, I construct and merge the paths below:

- 1st orthant
  - Fixed Path: (0, 0), (0, 1), ..., (0, 64)
  - Hamiltonian Path: (0, 64) ==> (0, 128)
- 2nd orthant
  - Hamiltonian Path: (0, 128) ==> (-128, 0)
- 3rd orthant
  - Hamiltonian Path: (-128, 0) ==> (0, -128)
- 4th orthant
  - Fixed Path: (0, -128), (1, -128), ..., (128, -128)
  - Hamiltonian Path: (128, -128) ==> (0, -64)
  - Fixed Path: (0, -64), (0, -63), ..., (0, 0)

Hamiltonian Paths are generated with [LKH-3](http://webhotel4.ruc.dk/~keld/research/LKH-3/).

### Key Ideas

#### Fixed Paths

- Fixed path in 4th orthant help us to return the initial configuration with small configuration costs

#### Edges between Pixels

In the discussion, we assumed only steps with distance 1, that is, direction of {(0, 1), (1, 0), (-1, 0), (0, -1)}.
However we could use steps of distance greater than 1 under some conditions:

- Diagonal steps {(1, 1), (-1, 1), (-1, -1), (1, -1)} are always available
- Moving the arms {32} and {16, 8, 4, 2, 1, 1} alternately, we could jump two pixels along x axis or y axis

## Download Dataset

```shell
kaggle competitions download -c santa-2022 -p data
unzip data/santa-2022.zip -d data
```

## Installation

```shell
poetry install
poetry shell
```

## Download and Build LKH-3

The command below generates `bin/LKH`.

```shell
inv lkh-build
```

## Solve Hamiltonian Path Problems

```shell
inv hpp-orthant1
```

Similarly, `inv hpp-orthant2` `inv hpp-orthant3` `inv hpp-orthant4` work.

## Generate Submission

```shell
inv merge-tours \
  --orthant1 outputs/orthant1-18065941.tour \
  --orthant2 outputs/orthant2-18532597.tour \
  --orthant3 outputs/orthant3-18458619.tour \
  --orthant4 outputs/orthant4-18674905.tour
```
