Lagrangian SAT
==============

A neural network framework for solving SAT (Boolean satisfiability) problems and optimization problems using TensorFlow. This project implements various models, including neural networks and continuous-time recurrent neural networks (CTRNN).

Table of Contents
-----------------
- Installation
- Usage
- Project Structure
- Models
- Loss Functions
- Testing
- Contributing
- License

Installation
------------

To install the required dependencies, you can use pip. Make sure you have Python 3.x installed.

.. code-block:: bash

    pip install tensorflow ncps

Usage
-----

You can generate SAT problems and train the models using the provided scripts. Here is a basic example of how to train a model:

.. code-block:: python

    from train import train_sat_solver

    NUM_VARIABLES = 10
    NUM_CLAUSES = 50
    BATCH_SIZE = 32
    EPOCHS = 1000

    trained_model = train_sat_solver(NUM_VARIABLES, NUM_CLAUSES, BATCH_SIZE, EPOCHS)

Project Structure
-----------------

The project is organized into several modules:

- **generator.py**: Contains functions to generate random SAT problems and optimization problems.
- **convertor.py**: Handles the conversion between SAT and optimization formulations.
- **lloss.py**: Defines the Lagrangian loss function used for training.
- **model.py**: Contains the definitions for various neural network models including SATSolver, OptimSolver, and CfCSolver.
- **train.py**: The main script to generate data and train the models.

Models
------

### SATSolver

A neural network model specifically designed for solving SAT problems using a bidirectional GRU architecture.

### OptimSolver

A neural network model tailored for optimization problems, also utilizing a bidirectional GRU architecture.

### CfCSolver

A model that leverages the `CfC` architecture from the `ncps` library to handle both SAT and optimization problems based on specified parameters during initialization.

Loss Functions
--------------

The project includes a custom Lagrangian loss function that adapts to both SAT and optimization problems. This loss function is defined in `lloss.py`.

Testing
-------

To run the unit tests for the models, simply execute:

.. code-block:: bash

    python -m unittest discover -s tests

Make sure you have created a `tests` directory containing your test files.

Contributing
------------

Contributions are welcome! Please feel free to submit issues or pull requests. Make sure to follow the coding standards and include tests for new features.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

