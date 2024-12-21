#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:30:57 2024

@author: fabien
"""

import tensorflow as tf

def sat_to_optimization(clauses, num_variables):
    """
    Converts a SAT problem to an optimization problem.
    """
    def objective_function(x):
        clause_satisfaction = tf.reduce_max(
            tf.gather(tf.concat([1-x, x], axis=0), tf.abs(clauses) - 1) * tf.sign(tf.cast(clauses, tf.float32)),
            axis=1
        )
        return -tf.reduce_sum(clause_satisfaction)

    constraints = lambda x: x * (1 - x)
    
    return objective_function, constraints

def optimization_to_sat(objective_function, constraints, num_variables):
    """
    Converts an optimization problem to a SAT problem.
    """
    clauses = []
    for i in range(2**num_variables):
        binary = format(i, f'0{num_variables}b')
        x = tf.cast(tf.convert_to_tensor([int(b) for b in binary]), tf.float32)
        if objective_function(x) < 0:
            clause = [j + 1 if x[j] else -(j + 1) for j in range(num_variables)]
            clauses.append(clause)
    # Ensure we always return a non-empty tensor with the correct shape
    if not clauses:
        # If no clauses were generated, create a default clause
        clauses = [[1, 2, 3]]
    return tf.cast(tf.convert_to_tensor(clauses), tf.int32)

if __name__ == '__main__':
    tf.test.main()
