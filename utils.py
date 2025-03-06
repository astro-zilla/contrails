import dataclasses
import json

import numpy as np
try:
    from pint import Quantity


    class AdvancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, Quantity):
                # return f"{o:~P}"
                return o.to_base_units().magnitude
            return super().default(o)

except ModuleNotFoundError:
    print("[contrails.utils] pint not installed, AdvancedJSONEncoder will not be available.")


def dataclass_from_dict(cls, dct):
    if dataclasses.is_dataclass(cls):
        fieldtypes = {field.name: field.type for field in dataclasses.fields(cls)}
        return cls(**{field: dataclass_from_dict(fieldtypes[field], dct[field]) for field in dct})
    else:
        return dct


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
    mu: float
    Y_h2o: float
    vx: float

    mdot: float
    BPR: float

    h0_bypass: float
    mu_bypass: float

    h0_core: float
    mu_core: float
    Y_h2o_core: float

    p_fan: float

    bl_nacelle_bypass: BoundaryLayer
    bl_bypass_core: BoundaryLayer
    bl_core_tail: BoundaryLayer
    bl_wake: BoundaryLayer


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


def cut_te(upper, lower, thickness):
    while abs(upper[-2].y - lower[-2].y).to_base_units().magnitude < thickness.to_base_units().magnitude:
        if upper[-2].x > lower[-2].x:
            del upper[-2]
        else:
            del lower[-2]

    u2, u1 = upper[-2:]
    l2, l1 = lower[-2:]
    if u2.x > l2.x:
        interp_var = (l1.x - u2.x) / (l1.x - l2.x)
        lower[-1].x = upper[-2].x
        lower[-1].y = interp_var * l2.y + (1 - interp_var) * l1.y
        upper[-1] = lower[-1]
        return 0
    else:
        interp_var = (u1.x - l2.x) / (u1.x - u2.x)
        upper[-1].x = lower[-2].x
        upper[-1].y = interp_var * u2.y + (1 - interp_var) * u1.y
        lower[-1] = upper[-1]
        return 1


def move_pt(line, seg_idx, pos):
    delta = pos - line[seg_idx].start
    line[seg_idx - 1].end += delta
    line[seg_idx - 1].control2 += delta
    line[seg_idx].start += delta
    line[seg_idx].control1 += delta


def move_path(path, delta):
    for seg in path:
        seg.start += delta
        seg.control1 += delta
        seg.control2 += delta
        seg.end += delta
