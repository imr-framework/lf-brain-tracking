import numpy as np

ref_file = "training_HF_reference.npz"

data = np.load(ref_file)

print("Keys:", data.files)

for key in data.files:
    arr = data[key]
    print(
        key,
        "shape =", arr.shape,
        "dtype =", arr.dtype,
        "min =", np.min(arr),
        "max =", np.max(arr)
    )

import numpy as np

d = np.load("training_HF_reference.npz")

q = d["reference_quantiles"]

for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    idx = int(p / 100 * 1000)
    print(f"P{p:3d}: {q[idx]:.4f}")