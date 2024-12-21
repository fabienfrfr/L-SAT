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
    Neural network model for 3-SAT problems.
    """
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.clause_dense = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.clause_dense(inputs)  # Encode clauses
        x = self.attention(x, x)      # Apply attention across clauses
        return self.output_layer(x)   # Output a value per clause

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
    solver = LNNSolver()
    data = tf.random.normal((2, 4, 3))  # Batch of size 2, 4 clauses per instance, 3 literals per clause
    output = solver(data)
    print("Output:", output.numpy())
