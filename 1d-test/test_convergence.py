"""
Quick test to diagnose convergence issue.
"""

import numpy as np
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd.boundary import SubsonicInletBC, SubsonicOutletBC

# Gas properties
gas = GasProperties(gamma=1.4, R=287.0)

# Simple nozzle
area_func = lambda x: 1.0 - 0.2 * np.sin(np.pi * x)
mesh = Mesh1D.uniform(0.0, 1.0, 100, area_func)

# Solver with LOWER CFL for stability
config = SolverConfig(
    cfl=0.5,  # Reduced from 0.8 for better stability
    max_iter=5000,
    convergence_tol=1e-6,
    print_interval=500,
    time_scheme='rk2'
)
solver = Solver1D(mesh, gas, n_scalars=2, config=config)

# Boundary conditions
p0 = 101325.0
T0 = 300.0
p_exit = 95000.0
Y_inlet = np.array([0.01, 0.005])

bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=Y_inlet)
bc_right = SubsonicOutletBC(p_exit=p_exit)
solver.set_boundary_conditions(bc_left, bc_right)

# Initial condition
M_init = 0.3
T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
rho_init = p_init / (gas.R * T_init)
u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

rho = np.full(100, rho_init)
u = np.full(100, u_init)
p = np.full(100, p_init)
Y = np.zeros((2, 100))
Y[0, :] = Y_inlet[0]
Y[1, :] = Y_inlet[1]

initial_state = FlowState(rho=rho, u=u, p=p, Y=Y, gas=gas)
solver.set_initial_condition(initial_state)

# Solve WITHOUT source terms
print("Testing convergence with improved residual and CFL=0.5...")
result = solver.solve()

if result['converged']:
    print(f"\n✓ SUCCESS: Converged in {result['iterations']} iterations")
    print(f"  Final residual: {result['final_residual']:.2e}")
else:
    print(f"\n✗ FAILED: Did not converge after {result['iterations']} iterations")
    print(f"  Final residual: {result['final_residual']:.2e}")
    print("\nResidual history (last 10):")
    for i, res in enumerate(solver.residual_history[-10:], 1):
        print(f"  {len(solver.residual_history)-10+i}: {res:.4e}")
