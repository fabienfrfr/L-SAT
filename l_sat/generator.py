#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:13:55 2024
@author: fabien
"""
import tensorflow as tf
#import unittest

def generate_sat_problem(num_variables, num_clauses):
    """
    Generates a random SAT problem using TensorFlow.
    """
    variables = tf.random.uniform((num_clauses, 3), minval=1, maxval=num_variables+1, dtype=tf.int32)
    signs = tf.random.uniform((num_clauses, 3), minval=0, maxval=2, dtype=tf.int32) * 2 - 1
    clauses = variables * signs
    return clauses

def generate_optimization_problem(num_variables, num_constraints):
    """    
    Generates a random optimization problem under constraints using TensorFlow.
    """
    c = tf.random.uniform((num_variables,))
    objective_function = lambda x: tf.reduce_sum(c * x)
    
    a = tf.random.uniform((num_constraints, num_variables))
    b = tf.random.uniform((num_constraints,))
    constraints = lambda x: b - tf.linalg.matvec(a, x)
    
    return objective_function, constraints

if __name__ == '__main__':
    tf.test.main()

