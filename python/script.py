#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

from networkx import read_gpickle
from pyomo.environ import ConcreteModel, Param, Set, Var, Constraint, Objective, NonNegativeReals, Reals, minimize, Suffix
from pyomo.opt import SolverFactory, ProblemFormat

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

model.delta = Param(model.N, model.E, initialize=temp, mutable=True)

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
    return model.c[i] * pressure_diff(model, i) + model.phi[i] * abs(model.ref_phi[i]) == 0

def gas_flow_conic(model, i):
    return model.c[i] * pressure_diff(model, i) + model.phi[i] ** 2 <= 0

def gas_flow(model, i):
    return model.c[i] * pressure_diff(model, i) + model.phi[i] * abs(model.phi[i]) <= 0

model.gas_flow_linearized = Constraint(model.P, rule=gas_flow_linearized)
model.gas_flow_conic = Constraint(model.P, rule=gas_flow_conic)
model.gas_flow = Constraint(model.P, rule=gas_flow)

model.gas_flow_conic.deactivate()
model.gas_flow.deactivate()

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

# Write .lp

DIR = 'products/'

model.write(filename=DIR + 'linear_model.lp',
    format=ProblemFormat.cpxlp,
    io_options={'symbolic_solver_labels': True}
)

#####
# 3 #
#####

instance = model.create_instance()

opt = SolverFactory('gurobi')
results = opt.solve(instance, tee=True, keepfiles=False)

## Write

os.makedirs(DIR, exist_ok=True)

instance.pprint(filename=DIR + 'linear_full.txt')
instance.display(filename=DIR + 'linear_sol.txt')

with open(DIR + 'linear_dual.txt', 'w') as f:
    for c in instance.component_objects(Constraint, active=True):
        for index in c:
            temp = instance.dual[c[index]]
            if temp is not None and temp != 0:
                print(c, '[{:d}]'.format(index), temp, file=f)

#####
# 6 #
#####

model.reconstruct()

for i in model.P:
    sign = np.sign(instance.phi[i].value)
    for j in model.N:
        model.delta[j, i] *= sign

model.gas_flow_linearized.deactivate()
model.gas_flow_conic.activate()

instance = model.create_instance()

results = opt.solve(instance, tee=True, keepfiles=False)

## Write

os.makedirs(DIR, exist_ok=True)

instance.pprint(filename=DIR + 'conic_full.txt')
instance.display(filename=DIR + 'conic_sol.txt')

with open(DIR + 'conic_dual.txt', 'w') as f:
    for c in instance.component_objects(Constraint, active=True):
        for index in c:
            temp = instance.dual[c[index]]
            if temp is not None and temp != 0:
                print(c, '[{:d}]'.format(index), temp, file=f)

#####
# 7 #
#####

model.gas_flow_conic.deactivate()
model.gas_flow.activate()

instance = model.create_instance()

opt = SolverFactory('ipopt')
results = opt.solve(instance, tee=True, keepfiles=False)

## Write

os.makedirs(DIR, exist_ok=True)

instance.pprint(filename=DIR + 'ipopt_full.txt')
instance.display(filename=DIR + 'ipopt_sol.txt')

with open(DIR + 'ipopt_dual.txt', 'w') as f:
    for c in instance.component_objects(Constraint, active=True):
        for index in c:
            temp = instance.dual[c[index]]
            if temp is not None and temp != 0:
                print(c, '[{:d}]'.format(index), temp, file=f)
