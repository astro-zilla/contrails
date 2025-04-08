import dataclasses
import json

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class AdvancedJSONEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from pint import Quantity
            self.quantity = Quantity
        except ModuleNotFoundError:
            self.quantity = NotImplemented

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, self.quantity):
            return o.to_base_units().magnitude
        return super().default(o)


@dataclasses.dataclass
class BoundaryLayer:
    y0: float
    n: int
    GR: float


@dataclasses.dataclass
class BoundaryCondition:
    M: float
    alpha: float
    T: float
    h0: float
    p: float
    p0: float
    nu: float
    Y_h2o: float
    vx: float

    mdot: float
    BPR: float
    Fnet: float

    h0_bypass: float
    nu_bypass: float
    A_bypass: float

    h0_core: float
    nu_core: float
    A_core: float
    Y_h2o_core: float

    p_fan: float
    A_fan: float

    bl_nacelle_bypass: BoundaryLayer
    bl_bypass_core: BoundaryLayer
    bl_core_tail: BoundaryLayer
    bl_wake: BoundaryLayer

    def __post_init__(self):
        if isinstance(self.bl_nacelle_bypass, dict):
            self.bl_nacelle_bypass = BoundaryLayer(**self.bl_nacelle_bypass)
        if isinstance(self.bl_bypass_core, dict):
            self.bl_bypass_core = BoundaryLayer(**self.bl_bypass_core)
        if isinstance(self.bl_core_tail, dict):
            self.bl_core_tail = BoundaryLayer(**self.bl_core_tail)
        if isinstance(self.bl_wake, dict):
            self.bl_wake = BoundaryLayer(**self.bl_wake)


def boundary_layer_mesh_stats(rho, V, mu, L, x, yplus, GR):
    # Reynolds number for leading edge y+ calculation (critical)
    # L should be LE radius for an aerofoil
    Re_L = rho * V * L / mu
    # Schlichting 1979 turbulent BL thickness approx for a flat plate using LE Re approx.
    C_f = 0.25 * 0.0576 * Re_L ** (-1 / 5)
    # compute first layer height
    tau_w = 0.5 * C_f * rho * V ** 2
    u_tau = np.sqrt(tau_w / rho)
    y0 = yplus * mu / rho / u_tau

    # Reynolds number for calculating final boundary layer height at TE
    Re_x = rho * V * x / mu
    delta_x = (0.37 * x * Re_x ** (-1 / 5))
    n = np.log(1 + (delta_x / y0) * (GR - 1)) / np.log(GR)

    final_layer_height = y0 * GR ** (n - 1)

    print(f"""y0 \t\t\t = {y0.to('mm'): .5f~P}
BL height    = {delta_x.to('mm'):0= 3g~P}
prism layers = {int(n): d}
last layer   = {final_layer_height.to('mm'): 3g~P}""")

    return BoundaryLayer(y0, int(n), GR)
