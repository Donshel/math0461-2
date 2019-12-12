#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

from networkx import read_gpickle
from pyomo.environ import ConcreteModel, Param, Set, Var, Constraint, Objective, NonNegativeReals, Reals, minimize, Suffix
from pyomo.opt import SolverFactory

# Function

def square(d):
    return dict([(key, value ** 2) for (key, value) in d.items()])

# Data

network = read_gpickle('test_network.instance')

# Model Object Creation

model = ConcreteModel()

# Parameters Definition

## Nodes
model.N = Set(initialize=network.ref_nodes)

model.d = Param(model.N, initialize=network.nodal_demands)
model.v = Param(model.N, initialize=network.value_unserved_demand)
model.Pi_m = Param(model.N, initialize=square(network.minimum_pressure_bounds))
model.Pi_p = Param(model.N, initialize=square(network.maximum_pressure_bounds))

Psi_m = network.minimum_nodal_injections
Psi_p = network.maximum_nodal_injections

for key in model.N:
    if model.d[key] != 0:
        Psi_m[key] = -model.d[key]
        Psi_p[key] = 0

model.Psi_m = Param(model.N, initialize=Psi_m)
model.Psi_p = Param(model.N, initialize=Psi_p)

## Edges
model.P = Set(initialize=network.ref_pipes.keys())
model.C = Set(initialize=network.ref_compressors.keys())
model.E = model.P | model.C

model.c = Param(model.E, initialize=network.friction_coefficients)
model.rho_m = Param(model.E, initialize=square(network.minimum_pressure_ratio))
model.rho_p = Param(model.E, initialize=square(network.maximum_pressure_ratio))
model.kappa = Param(model.E, initialize=network.compression_costs)

## Incidence

temp = {}
for (i, j), value in np.ndenumerate(network.incidence_matrix):
    temp[i + 1, j + 1] = value

model.delta = Param(model.N, model.E, initialize=temp)

## Reference flows

model.ref_phi = Param(model.E, initialize=network.reference_flows)

# Variables Definition

model.pi = Var(model.N, within=NonNegativeReals)
model.psi = Var(model.N, within=Reals)
model.phi = Var(model.E, within=Reals)

# Constraints Definition

def pressure_diff(model, i):
    return sum(model.pi[j] * model.delta[j, i] for j in model.N)

def gas_flow_linearized(model, i):
    return model.c[i] * pressure_diff(model, i) + (2 * model.phi[i] - model.ref_phi[i]) * abs(model.ref_phi[i]) == 0

model.gas_flow = Constraint(model.P, rule=gas_flow_linearized)

def operational_1(model, i):
    return sum(model.pi[j] * (-model.rho_m[i] if model.delta[j, i] == -1 else model.delta[j, i]) for j in model.N) >= 0

def operational_2(model, i):
    return sum(model.pi[j] * (-model.rho_p[i] if model.delta[j, i] == -1 else model.delta[j, i]) for j in model.N) <= 0

def operational_3(model, i):
    return model.phi[i] >= 0

model.operational_1 = Constraint(model.C, rule=operational_1)
model.operational_2 = Constraint(model.C, rule=operational_2)
model.operational_3 = Constraint(model.C, rule=operational_3)

def security_1(model, i):
    return model.pi[i] >= model.Pi_m[i]

def security_2(model, i):
    return model.pi[i] <= model.Pi_p[i]

model.security_1 = Constraint(model.N, rule=security_1)
model.security_2 = Constraint(model.N, rule=security_2)

def injection(model, i):
    return model.psi[i] == -sum(model.phi[j] * model.delta[i, j] for j in model.E)

model.injection = Constraint(model.N, rule=injection)

def contractual_1(model, i):
    return model.psi[i] >= model.Psi_m[i]

def contractual_2(model, i):
    return model.psi[i] <= model.Psi_p[i]

model.contractual_1 = Constraint(model.N, rule=contractual_1)
model.contractual_2 = Constraint(model.N, rule=contractual_2)

# Objective Definition

def objective(model):
    temp = sum(model.kappa[i] * pressure_diff(model, i) for i in model.C)
    temp += sum(model.v[i] * (model.d[i] + model.psi[i]) for i in model.N)
    return temp

model.objective = Objective(rule=objective, sense=minimize)

# Dual

model.dual = Suffix(direction=Suffix.IMPORT)

# Solve

opt = SolverFactory("gurobi")
results = opt.solve(model, tee=True, keepfiles=False)

## Write

DIR = 'products/txt/'
os.makedirs(DIR, exist_ok=True)

model.pprint(filename=DIR + 'linear_full.txt')

model.display(filename=DIR + 'linear_sol.txt')

with open(DIR + 'linear_dual.txt', 'w') as f:
    for c in model.component_objects(Constraint, active=True):
        for index in c:
            temp = model.dual[c[index]]
            if temp is not None and temp != 0:
                print(c, '[{:d}]'.format(index), temp, file=f)
