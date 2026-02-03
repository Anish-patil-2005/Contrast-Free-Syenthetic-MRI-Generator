import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

# Path to your file
FILE_PATH = "./data/raw/test/Mets_111/bravo.nii.gz" 

# 1. Load the NIfTI
img = nib.load(FILE_PATH)

# 2. Create a "Clean" 3D View
# 'display_mode=mosaic' shows the brain slices like a photo gallery
# 'threshold' removes the scary "noise" and only shows the brightest parts
print("📊 Generating a clean 3D slice gallery...")
plotting.plot_stat_map(
    img, 
    display_mode='z',      # 'z' shows Axial slices (the ones you know)
    cut_coords=10,         # Show 10 different slices across the brain
    title="Tumor Detection Gallery",
    threshold='auto',      # Automatically hides the "scary" noise
    cmap='cold_hot'        # Tumors will appear bright RED
)

# 3. Save as a simple image so you can look at it easily
plt.savefig("clean_tumor_view.png")
print("✅ Saved as 'clean_tumor_view.png'. Open this file to see the results!")
plt.show()