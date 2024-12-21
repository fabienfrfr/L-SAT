#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:57:42 2024
@author: fabien
"""

import tensorflow as tf
#from ncps.tf import CfC  # Just test

class LNNSolver(tf.keras.Model):
    """
    Neural network model for SAT problems.
    """
    def __init__(self, hidden_units=[64, 32]):
        super(LNNSolver, self).__init__()
        self.bidirectional_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.hidden_layers = [tf.keras.layers.Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.bidirectional_gru(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class LCfCSolver(tf.keras.Model):
    """
    Neural network model using the CfC architecture for both SAT and optimization problems.
    (Doesn't work for last Keras version... not updated)
    """
    def __init__(self, problem_type='SAT', num_units=50):
        super(LCfCSolver, self).__init__()
        self.problem_type = problem_type
        self.cfc_model = CfC(num_units)  # Initialize the CfC model

    def call(self, inputs):
        return self.cfc_model(inputs)

if __name__ == '__main__':
    tf.test.main()
