from svgpathtools import Arc, CubicBezier, Line, Path, QuadraticBezier


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


def get_intersection(a1, b1, a2, b2):
    # a1 + b1 * t1 = a2 + b2 * t2
    # t1 = (a2 - a1) / (b1 - b2)
    # t2 = (a1 - a2) / (b2 - b1)
    # solve for t1 and t2
    denom = b1.real * b2.imag - b1.imag * b2.real
    if denom == 0:
        return None
    t1 = ((a2.real - a1.real) * b2.imag - (a2.imag - a1.imag) * b2.real) / denom
    t2 = -((a1.real - a2.real) * b1.imag - (a1.imag - a2.imag) * b1.real) / denom
    return t1, t2


def get_distances_to_normal_intersection(t1: float, t2: float, segment1: CubicBezier, segment2: CubicBezier):
    # get points on the segments
    a1 = segment1.point(t1)
    a2 = segment2.point(t2)
    b1 = segment1.normal(t1)
    b2 = segment2.normal(t2)

    # get intersection point
    tt1, tt2 = get_intersection(a1, b1, a2, b2)
    if tt1 is None or tt2 is None:
        return None

    # calculate distances (signed such that the intersection lies between the lines)
    d1 = abs(b1 * tt1) * -tt1 / abs(tt1)
    d2 = abs(b2 * tt2) * tt2 / abs(tt2)
    return d1, d2


def plot(segment):
    import matplotlib.pyplot as plt
    import numpy as np
    t = np.linspace(0, 1, 100)
    pts = segment.point(t)
    plt.plot(pts.real, -pts.imag)


def fillet(segment1: CubicBezier, segment2: CubicBezier, r, h, rwake, lwake):
    def f(t, segment1, segment2, r):
        t1, t2 = t
        d1, d2 = get_distances_to_normal_intersection(t1, t2, segment1, segment2)
        return [d1 - r, d2 - r]

    import scipy
    import numpy as np
    t1, t2 = scipy.optimize.root(f, x0=np.array([0, 0]), args=(segment1, segment2, r), method='lm').x
    i1, i2 = get_intersection(segment1.point(t1), segment1.normal(t1), segment2.point(t2), segment2.normal(t2))

    intersection = segment1.point(t1) + segment1.normal(t1) * i1

    segment1.start, segment1.control1, segment1.control2, segment1.end = segment1.cropped(0, t1).bpoints()
    segment2.start, segment2.control1, segment2.control2, segment2.end = segment2.cropped(0, t2).bpoints()

    fillet = Arc(start=segment1.end, radius=complex(r, r), rotation=0, large_arc=False, sweep=True, end=segment2.end)

    # find initial gradient of wake
    bwake = segment1.unit_tangent(1) + segment2.unit_tangent(1)
    bwake /= abs(bwake)

    wake_initial = Line(intersection - bwake * (h + r), intersection + bwake * (h + r))

    print(wake_initial, intersection, fillet)
    (tf, tw), = fillet.intersect(wake_initial)

    wake_initial = wake_initial.cropped(tw, 1)
    fillet1, fillet2 = fillet.split(tf)

    tmp_initial = Line(intersection, intersection + 2 * bwake * (rwake / bwake.imag))
    tmp_end = Line(complex(-lwake, -rwake), complex(lwake, -rwake))

    (q1, q2), = tmp_initial.intersect(tmp_end)
    wake_quad = QuadraticBezier(wake_initial.end, tmp_initial.point(q1), complex(2, -rwake))
    wake_end = Line(wake_quad.end, complex(lwake, -rwake))

    return fillet1, fillet2.reversed(), Path(wake_initial), Path(wake_quad, wake_end)
