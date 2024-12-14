#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:57:42 2024
@author: fabien
"""

import tensorflow as tf
#from ncps.tf import CfC  # Just test

class SATSolver(tf.keras.Model):
    """
    Neural network model for SAT problems.
    """
    def __init__(self, hidden_units=[64, 32]):
        super(SATSolver, self).__init__()
        self.bidirectional_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.hidden_layers = [tf.keras.layers.Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.bidirectional_gru(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class OptimSolver(tf.keras.Model):
    """
    Neural network model for optimization problems.
    """
    def __init__(self, hidden_units=[64, 32]):
        super(OptimSolver, self).__init__()
        self.bidirectional_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.hidden_layers = [tf.keras.layers.Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.bidirectional_gru(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class CfCSolver(tf.keras.Model):
    """
    Neural network model using the CfC architecture for both SAT and optimization problems.
    (Doesn't work for last Keras version... not updated)
    """
    def __init__(self, problem_type='SAT', num_units=50):
        super(CfCSolver, self).__init__()
        self.problem_type = problem_type
        self.cfc_model = CfC(num_units)  # Initialize the CfC model

    def call(self, inputs):
        return self.cfc_model(inputs)

# Unit tests
class TestNeuralNetworks(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.num_variables = 10
        self.num_clauses = 50
        self.batch_size = 32

    def test_sat_solver(self):
        model = SATSolver()
        inputs = tf.random.uniform((self.batch_size, self.num_clauses, self.num_variables))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_clauses, 1))
        self.assertTrue(tf.reduce_all(outputs >= 0) and tf.reduce_all(outputs <= 1))

    def test_optim_solver(self):
        model = OptimSolver()
        inputs = tf.random.uniform((self.batch_size, self.num_clauses, self.num_variables))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_clauses, 1))

if __name__ == '__main__':
    tf.test.main()
