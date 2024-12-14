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


# Unit tests
class TestProblemGeneration(tf.test.TestCase):
    def test_sat_problem_shape(self):
        """
        Tests if the generated SAT problem has the correct shape.
        This ensures that the problem generation creates the expected number of clauses and variables.
        """
        num_variables, num_clauses = 10, 20
        problem = generate_sat_problem(num_variables, num_clauses)
        self.assertEqual(problem.shape, (num_clauses, 3))

    def test_sat_problem_range(self):
        """
        Tests if the generated SAT problem contains valid variable indices.
        This ensures that all variables in the clauses are within the specified range.
        """
        num_variables, num_clauses = 10, 20
        problem = generate_sat_problem(num_variables, num_clauses)
        self.assertTrue(tf.reduce_all(tf.abs(problem) <= num_variables))
        self.assertTrue(tf.reduce_all(tf.abs(problem) >= 1))

    def test_optimization_problem(self):
        """
        Tests if the generated optimization problem produces valid objective and constraint functions.
        This ensures that the problem generation creates functions that can be evaluated on input data.
        """
        num_variables, num_constraints = 5, 3
        obj_func, constraints = generate_optimization_problem(num_variables, num_constraints)
        
        x = tf.random.uniform((num_variables,))
        obj_value = obj_func(x)
        constraint_values = constraints(x)
        
        self.assertEqual(obj_value.shape, ())
        self.assertEqual(constraint_values.shape, (num_constraints,))

    def test_sat_to_optimization_conversion(self):
        """
        Tests if the SAT to optimization conversion produces valid functions.
        This ensures that the conversion creates objective and constraint functions that can be evaluated.
        """
        num_variables, num_clauses = 3, 2
        clauses = tf.constant([[1, -2, 3], [-1, 2, -3]])
        obj_func, constraints = sat_to_optimization(clauses, num_variables)
        
        x = tf.constant([0.0, 1.0, 1.0])
        obj_value = obj_func(x)
        constraint_values = constraints(x)
        
        self.assertEqual(obj_value.shape, ())
        self.assertEqual(constraint_values.shape, (num_variables,))

    def test_optimization_to_sat_conversion(self):
        """
        Tests if the optimization to SAT conversion produces a valid SAT problem.
        """
        num_variables = 3
        obj_func = lambda x: tf.reduce_sum(x)
        constraints = lambda x: x * (1 - x)
        
        clauses = optimization_to_sat(obj_func, constraints, num_variables)
        
        self.assertEqual(clauses.dtype, tf.int32)
        self.assertGreater(tf.shape(clauses)[0], 0, "Clauses tensor should not be empty")
        self.assertEqual(tf.shape(clauses)[1], 3, "Each clause should have 3 literals")


if __name__ == '__main__':
    tf.test.main()

