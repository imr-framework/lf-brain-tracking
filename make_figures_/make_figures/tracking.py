import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# USER SETTINGS
# ==========================================================

BASE_FOLDER = "Data_prospective_study/Evaluate_prospective_refine_range/59081"

OUTPUT_FILE = "26184_longitudinal_final_recon_59081.png"


# Select slices independently

SLICE_SELECTION = {

    "visit1": {
        "3T": 37,
        "47mT": 20,
        "Recon2": 20
    },

    "visit2": {
        "3T": 55,
        "47mT": 40,
        "Recon2": 40
    },

    "visit3": {
        "3T": 51,
        "47mT": 25,
        "Recon2": 25
    }

}

SLICE_SELECTION = {

    "visit1": {
        "3T": 49,
        "47mT": 24,
        "Recon2": 24
    },

    "visit2": {
        "3T": 55,
        "47mT": 17,
        "Recon2": 17
    },

    "visit3": {
        "3T": 51,
        "47mT": 25,
        "Recon2": 25
    },

    "visit4": {
            "3T": 51,
            "47mT": 25,
            "Recon2": 25
        }

}

SLICE_DIRECTION = {

    "3T": "lower",
    "47mT": "upper",
    "Recon2": "upper"

}

# 0 = sagittal
# 1 = coronal
# 2 = axial

AXIS = 2



# ==========================================================
# FUNCTIONS
# ==========================================================

def load_correct_orientation(path):

    img = nib.load(path)

    # Correct orientation only
    img = nib.as_closest_canonical(img)

    data = img.get_fdata()

    return data



def extract_slice(img, index, axis, direction="upper", name=None):

    max_slice = img.shape[axis] - 1

    if direction == "lower":
        index = max_slice - index


    if axis == 0:
        sl = img[index, :, :]

    elif axis == 1:
        sl = img[:, index, :]

    elif axis == 2:
        sl = img[:, :, index]

    else:
        raise ValueError("Axis must be 0,1,2")


    # K=1 rotates 90 degrees counter-clockwise
    sl = np.rot90(
        sl,
        k=1,
        axes=(0,1)
    )


    # left-right flip only for 3T
    if name == "3T":
        sl = np.fliplr(sl)


    return sl



# ==========================================================
# FIND VISITS
# ==========================================================

visits = sorted(
    [
        x for x in os.listdir(BASE_FOLDER)
        if x.startswith("visit")
        and os.path.isdir(
            os.path.join(BASE_FOLDER,x)
        )
    ]
)


print("Detected visits:")
for v in visits:
    print(v)



# ==========================================================
# LOAD DATA
# ==========================================================

images = []


for visit in visits:

    print("\nProcessing:", visit)

    visit_folder = os.path.join(
        BASE_FOLDER,
        visit
    )


    files = {

        "3T":
        "3T_image.nii.gz",

        "47mT":
        "47mT_image.nii.gz",

        "Recon2":
        "Recon2_image.nii.gz"

    }


    visit_images = {}


    for image_name, filename in files.items():

        path = os.path.join(
            visit_folder,
            filename
        )


        if not os.path.exists(path):

            raise FileNotFoundError(path)


        data = load_correct_orientation(path)


        slice_number = SLICE_SELECTION[visit][image_name]


        visit_images[image_name] = extract_slice(
            data,
            slice_number,
            AXIS,
            SLICE_DIRECTION[image_name],
            image_name
        )


    images.append(
        visit_images
    )



# ==========================================================
# DISPLAY
# ==========================================================

nrows = len(visits)
ncols = 3


fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(9,3*nrows)
)


if nrows == 1:
    axes = np.expand_dims(
        axes,
        axis=0
    )


columns = [
    "3T",
    "47mT",
    "Recon2"
]


titles = [
    "3 T",
    "47 mT",
    "Synth-recon"
]


for c in range(ncols):

    axes[0,c].set_title(
        titles[c],
        fontsize=14,
        pad=5
    )


for r,visit in enumerate(visits):

    for c,name in enumerate(columns):

        ax = axes[r,c]


        img = images[r][name]


        # RAW display, no normalization
        ax.imshow(
            img,
            cmap="gray",
            interpolation="nearest"
        )


        ax.axis("off")


    axes[r,0].text(
        -0.08,
        0.5,
        visit,
        rotation=90,
        transform=axes[r,0].transAxes,
        va="center",
        ha="center",
        fontsize=12
    )



plt.subplots_adjust(
    left=0,
    right=1,
    top=0.96,
    bottom=0,
    wspace=0,
    hspace=0
)


plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)


plt.show()


print("\nSaved:")
print(OUTPUT_FILE)