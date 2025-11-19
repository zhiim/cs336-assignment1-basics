from math import cos, pi


def cosine_learning_rate_schedule(
    t: int, lr_max: float, lr_min: float, t_w: int, t_c: int
):
    if t < t_w:
        lr_t = t / t_w * lr_max
    elif t > t_c:
        lr_t = lr_min
    else:
        lr_t = lr_min + 0.5 * (1 + cos((t - t_w) / (t_c - t_w) * pi)) * (
            lr_max - lr_min
        )

    return lr_t
