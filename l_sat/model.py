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
    Neural network model for solving SAT and optimization problems.
    """
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        # Define a sequential model for encoding input clauses or constraints
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),  # Dense layer for initial processing
            tf.keras.layers.MultiHeadAttention(num_heads, key_dim=hidden_dim),  # Attention layer
            tf.keras.layers.Dense(hidden_dim, activation='relu')  # Another Dense layer
        ])
        self.var_generator = tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for variable values

    def call(self, inputs, num_variables):
        x = self.encoder(inputs)  # Encode the input
        query = tf.ones((tf.shape(x)[0], num_variables, 1))  # Create a query tensor for attention
        attention = tf.nn.softmax(tf.matmul(query, x, transpose_b=True), axis=-1)  # Compute attention weights
        context = tf.matmul(attention, x)  # Apply attention to the encoded input
        return self.var_generator(context)  # Generate variable values

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
    output = solver(data, 3)
    print("Output:", output.numpy())
