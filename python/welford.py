import numpy as np


def oneline_update(x: float, n: int, s: float, a: float):
    """
    a: average of (n-1) x_i
    s: M_{2,n-1}
    """
    b = a + (x - a) / n
    s += (x - b) * (x - a)

    return s, b


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)

a: float = 0
s: float = 0

for i in range(arr.size):
    s, a = oneline_update(arr[i], i + 1, s, a)
    subarr = arr[0 : i + 1]
    print(f"ref     average = {subarr.mean()}, var = {subarr.var()}")
    print(f"welford average = {a}, var = {s/subarr.size}")
    print()
