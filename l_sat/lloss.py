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
        self.lambda_ = tf.Variable(lambda_init, trainable=True)

    def call(self, y_true, y_pred):
        if self.problem_type == 'SAT':
            return self.sat_loss(y_true, y_pred)
        elif self.problem_type == 'optimization':
            return self.optimization_loss(y_true, y_pred)
        else:
            raise ValueError("Invalid problem type. Must be 'SAT' or 'optimization'.")

    def sat_loss(self, clauses, x):
        clause_satisfaction = tf.reduce_prod(
            tf.reduce_max(
                tf.gather(tf.concat([1-x, x], axis=-1), tf.abs(clauses) - 1) * tf.sign(tf.cast(clauses, tf.float32)),
                axis=-1
            ),
            axis=-1
        )
        constraint = tf.reduce_sum((x * (1 - x))**2, axis=-1)
        return tf.reduce_mean((1 - clause_satisfaction)**2 + self.lambda_ * constraint)

    def optimization_loss(self, problem_data, x):
        obj_func, constraints = problem_data
        obj_value = obj_func(x)
        constraint_values = [c['fun'](x) for c in constraints]
        loss = obj_value + tf.reduce_sum([
            self.lambda_ * v**2 if c['type'] == 'eq' else
            self.lambda_ * tf.maximum(0.0, v)**2 for c, v in zip(constraints, constraint_values)
        ])
        return tf.reduce_mean(loss)

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
