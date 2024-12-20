#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:36:45 2024
@author: fabien
"""
import tensorflow as tf

class LagrangianLoss(tf.keras.losses.Loss):
    def __init__(self, problem_type, problem_data, lambda_init=1.0, name="lagrangian_loss"):
        super().__init__(name=name)
        self.problem_type = problem_type
        self.problem_data = problem_data
        self.lambda_ = tf.Variable(lambda_init, trainable=True)

    def call(self, y_true, y_pred):
        if self.problem_type == 'SAT':
            return self.sat_loss(y_pred)
        elif self.problem_type == 'optimization':
            return self.optimization_loss(y_pred)
        else:
            raise ValueError("Invalid problem type. Must be 'SAT' or 'optimization'.")

    def sat_loss(self, x):
        clauses = self.problem_data
        num_variables = tf.shape(x)[0]
        clause_satisfaction = tf.reduce_max(
            tf.gather(tf.concat([1-x, x], axis=0), tf.abs(clauses) - 1) * tf.sign(tf.cast(clauses, tf.float32)),
            axis=1
        )
        sat_loss = tf.reduce_sum(1.0 - clause_satisfaction)
        constraint_penalty = tf.reduce_sum(self.lambda_ * (x * (1 - x)))
        return sat_loss + constraint_penalty

    def optimization_loss(self, x):
        obj_func, constraints = self.problem_data
        obj_value = obj_func(x)
        constraint_values = [c['fun'](x) for c in constraints]
        constraint_penalty = tf.reduce_sum([
            self.lambda_ * tf.maximum(0.0, -v) if c['type'] == 'ineq' else
            self.lambda_ * tf.abs(v) for c, v in zip(constraints, constraint_values)
        ])
        return obj_value + constraint_penalty

if __name__ == '__main__':
    #tf.test.main()
    # Examples
    num_variables = 10
    num_clauses = 50
    # SAT
    clauses = generate_sat_problem(num_variables, num_clauses)
    sat_loss = LagrangianLoss('SAT', clauses)
    # Optim
    obj_func, constraints = generate_optimization_problem(num_variables, 5)
    optim_loss = LagrangianLoss('optimization', (obj_func, constraints))
    # Model & compile
    model = tf.keras.Sequential([tf.keras.layers.Dense(num_variables, activation='sigmoid', input_shape=(num_variables,))])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=sat_loss)  # or optim_loss
    # Train
    model.fit(x=tf.expand_dims(clauses, 0), y=tf.zeros((1, num_variables)), epochs=1000)
