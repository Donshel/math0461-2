#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

from networkx import read_gpickle
from pyomo.environ import ConcreteModel, Param, Set, Var, Constraint, Objective, NonNegativeReals, Reals, minimize, Suffix
from pyomo.opt import SolverFactory, ProblemFormat

from gurobipy import read

from matplotlib import rc
from matplotlib import pyplot as plt

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 12})

# Function

def square(d):
    return dict([(key, value ** 2) for (key, value) in d.items()])

#########################
# 1. Problem definition #
#########################

# Data

network = read_gpickle('resources/instance/test_network.instance')

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

def gas_flow(model, i):
    return model.c[i] * pressure_diff(model, i) + model.phi[i] * abs(model.phi[i]) == 0

def gas_flow_linear(model, i):
    return model.c[i] * pressure_diff(model, i) + model.phi[i] * abs(model.ref_phi[i]) == 0

def gas_flow_conic(model, i):
    return model.c[i] * pressure_diff(model, i) + model.phi[i] ** 2 <= 0

model.gas_flow = Constraint(model.P, rule=gas_flow)
model.gas_flow_linear = Constraint(model.P, rule=gas_flow_linear)
model.gas_flow_conic = Constraint(model.P, rule=gas_flow_conic)

model.gas_flow_linear.deactivate()
model.gas_flow_conic.deactivate()

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

####################
# 3. Linearization #
####################

model.gas_flow.deactivate()
model.gas_flow_linear.activate()

linear = model.clone()

opt = SolverFactory('gurobi')
results = opt.solve(linear, keepfiles=False)

DIR = 'products/'
os.makedirs(DIR, exist_ok=True)

linear.display(filename=DIR + 'linear_solultion.txt')

###############################################
# 4. Dual variables & 5. Sensitivity analysis #
###############################################

model.write(filename=DIR + 'linear_model.lp',
    format=ProblemFormat.cpxlp,
    io_options={'symbolic_solver_labels': True}
)

gmodel = read('products/linear_model.lp')
gmodel.optimize()

constrs = {'Name':[], 'Dual':[], 'Slack':[], 'Lower':[], 'Upper':[]}

for c in gmodel.getConstrs():
    constrs['Name'].append(c.ConstrName)
    constrs['Dual'].append(c.Pi)
    constrs['Slack'].append(c.Slack)
    constrs['Lower'].append(c.SARHSLow)
    constrs['Upper'].append(c.SARHSUp)

constrs = pd.DataFrame(constrs)

with open(DIR + 'linear_constraints.txt', 'w') as f:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(constrs, file=f)

#######################
# 6. Conic relaxation #
#######################

model.reconstruct()

for i in model.P:
    sign = np.sign(linear.phi[i].value)
    for j in model.N:
        model.delta[j, i] *= sign

model.gas_flow_linear.deactivate()
model.gas_flow_conic.activate()

conic = model.clone()

results = opt.solve(conic, keepfiles=False)

conic.display(filename=DIR + 'conic_solution.txt')

#################
# 7. Non linear #
#################

model.reconstruct()

for i in model.P:
    sign = np.sign(linear.phi[i].value)
    for j in model.N:
        model.delta[j, i] *= sign

model.gas_flow_conic.deactivate()
model.gas_flow.activate()

nonlinear = model.clone()

opt = SolverFactory('bin/ipopt', solver_io='nl')
results = opt.solve(nonlinear, keepfiles=False)

nonlinear.display(filename=DIR + 'nonlinear_solution.txt')

#########
# Plots #
#########

N = len(model.N)
E = len(model.E)

pi = np.zeros((N, 3))
psi = np.zeros((N, 3))
phi = np.zeros((E, 3))

for i in model.N:
    pi[i - 1, 0] = linear.pi[i].value
    pi[i - 1, 1] = conic.pi[i].value
    pi[i - 1, 2] = nonlinear.pi[i].value

    psi[i - 1, 0] = linear.psi[i].value
    psi[i - 1, 1] = conic.psi[i].value
    psi[i - 1, 2] = nonlinear.psi[i].value

for i in model.E:
    phi[i - 1, 0] = linear.phi[i].value
    phi[i - 1, 1] = conic.phi[i].value
    phi[i - 1, 2] = nonlinear.phi[i].value

dual = []

for c in gmodel.getConstrs():
    if 'injection' in c.ConstrName:
        dual.append(c.Pi)

dual = np.array(dual)

DIR = 'products/pdf/'

os.makedirs(DIR, exist_ok=True)

lineObjects = plt.plot(np.arange(1, N + 1), pi)
plt.legend(lineObjects, ('linear', 'conic', 'non-linear'))
plt.xlabel('$i$')
plt.ylabel('$\\pi_i$')
plt.xticks(np.arange(1, N + 1))
plt.grid()
plt.savefig(DIR + '{}.pdf'.format('pi'), bbox_inches='tight')
plt.close()

lineObjects = plt.plot(np.arange(1, N + 1), psi)
plt.legend(lineObjects, ('linear', 'conic', 'non-linear'))
plt.xlabel('$i$')
plt.ylabel('$\\psi_i$')
plt.xticks(np.arange(1, N + 1))
plt.grid()
plt.savefig(DIR + '{}.pdf'.format('psi'), bbox_inches='tight')
plt.close()

lineObjects = plt.plot(np.arange(1, E + 1), phi)
plt.legend(lineObjects, ('linear', 'conic', 'non-linear'))
plt.xlabel('$i$')
plt.ylabel('$\\phi_i$')
plt.xticks(np.arange(1, E + 1))
plt.grid()
plt.savefig(DIR + '{}.pdf'.format('phi'), bbox_inches='tight')
plt.close()

plt.bar(np.arange(1, dual.shape[0] + 1), dual)
plt.xlabel('$i$')
plt.ylabel('dual$_i$')
plt.xticks(np.arange(1, N + 1))
plt.savefig(DIR + '{}.pdf'.format('dual'), bbox_inches='tight')
plt.close()
