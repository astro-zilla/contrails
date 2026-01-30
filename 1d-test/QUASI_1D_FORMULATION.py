"""
Reference: Understanding Quasi-1D Nozzle Flow Discretization

From Anderson CFD textbook, the quasi-1D Euler equations are:

Conservative form:
    ∂/∂t(ρA) + ∂/∂x(ρuA) = 0                    [continuity]
    ∂/∂t(ρuA) + ∂/∂x((ρu² + p)A) = p·∂A/∂x      [momentum]
    ∂/∂t(ρEA) + ∂/∂x(ρuHA) = 0                  [energy]

Dividing by A:
    ∂/∂t(ρ) + (1/A)·∂/∂x(ρuA) = 0
    ∂/∂t(ρu) + (1/A)·∂/∂x((ρu² + p)A) = (p/A)·∂A/∂x
    ∂/∂t(ρE) + (1/A)·∂/∂x(ρuHA) = 0

In finite volume form with cell-averaged variables U_i = [ρ, ρu, ρE]:
    dU_i/dt = -(1/V_i)·[A_{i+1/2}·F_{i+1/2} - A_{i-1/2}·F_{i-1/2}] + S_i

where V_i = A_i·Δx is the cell volume, and:
    - F is the 1D Euler flux
    - S = [0, p·∂A/∂x/A, 0] is the geometric source term

The source term S_momentum = p/A · dA/dx is REQUIRED for momentum equation.
Without it, the solver treats area as constant locally.

Test this hypothesis by re-enabling the area source term.
"""
print(__doc__)

