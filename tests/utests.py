#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:13:55 2024
@author: fabien
"""
import tensorflow as tf

# Generation and convertion tests
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

# Solver tests
class TestModelConversion(tf.test.TestCase):
    def test_ortools_to_sat(self):
        """
        Tests the conversion from OR-Tools to SAT.
        """
        model = cp_model.CpModel()
        x = model.NewBoolVar('x')
        y = model.NewBoolVar('y')
        model.Add(x != y)
        
        sat_clauses = ortools_to_sat(model)
        
        self.assertIsInstance(sat_clauses, tf.Tensor)
        self.assertEqual(sat_clauses.dtype, tf.int32)
        self.assertGreater(tf.shape(sat_clauses)[0], 0)
    
    def test_pyomo_to_optimization(self):
        """
        Tests the conversion from Pyomo to optimization problem.
        """
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 1))
        model.y = pyo.Var(bounds=(0, 1))
        model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.maximize)
        model.con = pyo.Constraint(expr=model.x + model.y <= 1)
        
        obj_func, constraints = pyomo_to_optimization(model)
        
        self.assertIsInstance(obj_func, callable)
        self.assertIsInstance(constraints, list)
        self.assertGreater(len(constraints), 0)

# Model tests
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
        
# Loss Tests
class TestLagrangianLoss(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.num_variables = 5
        self.num_clauses = 10

    def test_sat_loss(self):
        clauses = tf.constant([[1, -2, 3], [-1, 2, -3], [2, 3, -4]], dtype=tf.int32)
        loss_fn = LagrangianLoss('SAT', clauses)
        x = tf.constant([0.1, 0.9, 0.8, 0.2, 0.5])
        loss_value = loss_fn(None, x)
        self.assertIsInstance(loss_value, tf.Tensor)
        self.assertEqual(loss_value.shape, ())

    def test_optimization_loss(self):
        obj_func = lambda x: tf.reduce_sum(x)
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 1 - tf.reduce_sum(x)},
            {'type': 'eq', 'fun': lambda x: x[0] - 0.5}
        ]
        loss_fn = LagrangianLoss('optimization', (obj_func, constraints))
        x = tf.constant([0.3, 0.2, 0.1, 0.2, 0.1])
        loss_value = loss_fn(None, x)
        self.assertIsInstance(loss_value, tf.Tensor)
        self.assertEqual(loss_value.shape, ())

    def test_invalid_problem_type(self):
        with self.assertRaises(ValueError):
            LagrangianLoss('invalid_type', None)

    def test_lambda_update(self):
        clauses = tf.constant([[1, -2, 3], [-1, 2, -3]], dtype=tf.int32)
        loss_fn = LagrangianLoss('SAT', clauses)
        x = tf.constant([0.1, 0.9, 0.8])
        with tf.GradientTape() as tape:
            loss_value = loss_fn(None, x)
        grads = tape.gradient(loss_value, [loss_fn.lambda_])
        self.assertIsNotNone(grads[0])

if __name__ == '__main__':
    tf.test.main()

