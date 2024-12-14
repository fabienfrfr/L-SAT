#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:34:56 2024
@author: fabien
"""

import tensorflow as tf
from ortools.sat.python import cp_model
import pyomo.environ as pyo

def ortools_to_sat(model):
    """
    Converts an OR-Tools model to SAT clauses.
    
    Args:
    model: OR-Tools model to convert.
    
    Returns:
    tf.Tensor: Tensor representing SAT clauses.
    """
    cp_model = cp_model.CpModel()
    variables = {}
    
    for var in model.variables():
        if var.lb == 0 and var.ub == 1:
            variables[var.name] = cp_model.NewBoolVar(var.name)
        else:
            variables[var.name] = cp_model.NewIntVar(var.lb, var.ub, var.name)
    
    for ct in model.constraints():
        expr = ct.expr()
        if isinstance(expr, cp_model.LinearExpr):
            cp_model.Add(expr)
        elif isinstance(expr, cp_model.BooleanExpr):
            cp_model.AddBoolOr(expr)
        # Add more constraint types as needed
    
    objective = model.objective()
    if objective.maximize:
        cp_model.Maximize(sum(coef * variables[var.name] for var, coef in objective.coefficients().items()))
    else:
        cp_model.Minimize(sum(coef * variables[var.name] for var, coef in objective.coefficients().items()))
    
    solver = cp_model.CpSolver()
    status = solver.Solve(cp_model)
    
    clauses = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for var_name, var in variables.items():
            if solver.Value(var) == 1:
                clauses.append([int(var_name)])
            else:
                clauses.append([-int(var_name)])
    
    return tf.constant(clauses, dtype=tf.int32)

def pyomo_to_optimization(model):
    """
    Converts a Pyomo model to an optimization problem.
    
    Args:
    model: Pyomo model to convert.
    
    Returns:
    tuple: (objective_function, constraints)
    """
    variables = list(model.component_data_objects(pyo.Var))
    
    def objective_function(x):
        return sum(model.objective[i].expr() for i in model.objective)
    
    constraints = []
    for constraint in model.component_data_objects(pyo.Constraint):
        if constraint.equality:
            constraints.append({'type': 'eq', 'fun': lambda x: constraint.body() - constraint.rhs})
        else:
            if constraint.lower is not None:
                constraints.append({'type': 'ineq', 'fun': lambda x: constraint.body() - constraint.lower})
            if constraint.upper is not None:
                constraints.append({'type': 'ineq', 'fun': lambda x: constraint.upper - constraint.body()})
    
    return objective_function, constraints

# Unit tests
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

if __name__ == '__main__':
    tf.test.main()
