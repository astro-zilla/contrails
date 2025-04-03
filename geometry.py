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
