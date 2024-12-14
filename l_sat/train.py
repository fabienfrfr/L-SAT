#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:01:04 2024
@author: fabien
"""
import tensorflow as tf
from generator import generate_sat_problem
from generator import sat_to_optimization
from lloss import LagrangianLoss
from model import SATSolver

def generate_batch(batch_size, num_variables, num_clauses):
    """Generate a batch of SAT problems."""
    batch = []
    for _ in range(batch_size):
        clauses = generate_sat_problem(num_variables, num_clauses)
        batch.append(clauses)
    return tf.stack(batch)

def train_sat_solver(num_variables, num_clauses, batch_size=32, epochs=100):
    # Initialize model
    model = SATSolver()
    # Initialize loss function
    loss_fn = LagrangianLoss('SAT', None)  # We'll update problem_data for each batch
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss_fn.problem_data = x  # Update problem_data for the current batch
            loss_value = loss_fn(None, predictions)
        
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value
    
    for epoch in range(epochs):
        # Generate a new batch of SAT problems
        batch = generate_batch(batch_size, num_variables, num_clauses)
        # Perform a training step
        loss_value = train_step(batch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")
    return model

if __name__ == "__main__":
    NUM_VARIABLES = 10
    NUM_CLAUSES = 50
    BATCH_SIZE = 32
    EPOCHS = 1000
    
    trained_model = train_sat_solver(NUM_VARIABLES, NUM_CLAUSES, BATCH_SIZE, EPOCHS)
    print("Training completed.")
    
    # Optional: Test the model on a new SAT problem
    test_problem = generate_sat_problem(NUM_VARIABLES, NUM_CLAUSES)
    solution = trained_model(tf.expand_dims(test_problem, 0))
    print("Test problem solution:", solution.numpy())
