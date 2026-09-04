import numpy as np

file = r"Data_prospective_study/t2_sim/AFRICAN.GREEN_NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000/AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_1x1x2mm.npy"


arr = np.load(
    file,
    allow_pickle=True
)

print("Loaded object type:")
print(type(arr))

print("\nArray shape:")
print(arr.shape)

print("\nArray dtype:")
print(arr.dtype)


# If it is a 0-D object, unwrap it
if arr.shape == ():
    obj = arr.item()

    print("\nAfter .item():")
    print("Type:", type(obj))

    if isinstance(obj, (list, tuple)):
        print("Length:", len(obj))

        for i, x in enumerate(obj):
            print(
                f"\nIndex {i}:"
            )
            print("Type:", type(x))
            print("Shape:", getattr(x, "shape", None))

    elif isinstance(obj, dict):

        print("Keys:")
        print(obj.keys())

        for key, value in obj.items():
            print(
                f"\nKey {key}:"
            )
            print("Type:", type(value))
            print("Shape:", getattr(value, "shape", None))

else:

    print("\nDirect array:")
    print("Shape:", arr.shape)