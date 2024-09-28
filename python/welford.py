from typing import Tuple

import numpy as np


def welford_online_update(x: float, count: int, s: float, a: float) -> Tuple[float, float]:
    """
    a: average of x_i
    s: M_{2,n}
    """
    b = a + (x - a) / count
    s += (x - b) * (x - a)

    return s, b


def calculate_mean_and_var(arr: np.ndarray) -> Tuple[float, float]:
    a: float = 0
    s: float = 0

    for i in range(arr.size):
        s, a = welford_online_update(arr[i], i + 1, s, a)

    return a, s / arr.size


if __name__ == "__main__":

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)

    a, s = calculate_mean_and_var(arr)

    print(f"ref     average = {arr.mean()}, var = {arr.var()}")
    print(f"welford average = {a}, var = {s}")
    print()
