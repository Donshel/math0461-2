from gurobipy import read

# Parameters

targets = set(['c_l_security_1(16)_'])

# Model

model = read('products/linear_model.lp')
model.optimize()

# Print

print('Constraint', 'Shadow Price', 'Slack', 'Lower Range', 'Higher Range')
for c in model.getConstrs():
    if(c.ConstrName in targets):
        print(c.ConstrName, c.Pi, c.Slack, c.SARHSLow, c.SARHSUp)
