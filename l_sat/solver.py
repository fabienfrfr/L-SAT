#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:34:56 2024
@author: fabien
"""

import tensorflow as tf
from ortools.sat.python import cp_model
import pyomo.environ as pyo
import pulp

def ortools_to_sat(model):
    """
    Converts an OR-Tools model to SAT clauses.
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


def pulp_to_optimization(model):
    """
    Converts a PuLP model into a TensorFlow optimization problem.
    """
    variables = []
    lower_bounds = []
    upper_bounds = []
    
    for v in model.variables():
        variables.append(v.name)
        lower_bounds.append(v.lowBound if v.lowBound is not None else -float('inf'))
        upper_bounds.append(v.upBound if v.upBound is not None else float('inf'))
    
    tf_vars = tf.Variable(tf.zeros(len(variables)), dtype=tf.float32)
    
    def objective_function():
        return sum(c * tf_vars[variables.index(v.name)] for v, c in model.objective.items())
    
    constraints = []
    for name, constraint in model.constraints.items():
        if constraint.sense == pulp.LpConstraintLE:
            constraints.append(lambda: sum(c * tf_vars[variables.index(v.name)] for v, c in constraint.items()) <= -constraint.constant)
        elif constraint.sense == pulp.LpConstraintGE:
            constraints.append(lambda: sum(c * tf_vars[variables.index(v.name)] for v, c in constraint.items()) >= -constraint.constant)
        elif constraint.sense == pulp.LpConstraintEQ:
            constraints.append(lambda: sum(c * tf_vars[variables.index(v.name)] for v, c in constraint.items()) == -constraint.constant)
    
    return tf_vars, objective_function, constraints, lower_bounds, upper_bounds

# Example usage
if __name__ == '__main__':
    # Create a simple PuLP model
    model = pulp.LpProblem("Simple", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)
    model += x + 2*y
    model += x + y >= 3
    model += 2*x + y <= 10
    
    # Convert to TensorFlow format
    tf_vars, obj_func, constraints, lb, ub = pulp_to_optimization(model)
    
    # Use the TensorFlow optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            objective = obj_func()
            for constraint in constraints:
                objective += tf.maximum(constraint(), 0) * 1000  # Penalty for violated constraints
        gradients = tape.gradient(objective, [tf_vars])
        optimizer.apply_gradients(zip(gradients, [tf_vars]))
    
    for _ in range(1000):
        train_step()
    
    print("Solution:", tf_vars.numpy())

