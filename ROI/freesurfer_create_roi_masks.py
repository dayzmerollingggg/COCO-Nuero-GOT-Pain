#using freesurfer labeled sections from neurosynth tstat input to convert into npy ROI masks and plot

import os
import glob
import re
import numpy as np
import nibabel as nib
from nibabel.freesurfer import read_label
from nilearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from brainplotlib import brain_plot

# --- 1. Path Configuration ---
LABEL_DIR = "/mnt/labdata/got_project/test/ROI_analysis/got_pain_rois"
OUTPUT_MASK_DIR = "/mnt/labdata/got_project/test/ROI_analysis/9_3_freesurfer_create_roi_masks"
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_MASK_DIR, "brain_plots")

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# fsaverage5 standard mesh contains exactly 10,242 vertices per hemisphere
N_VERTICES = 10242

# --- 2. Build Curvature Underlay for brain_plot ---
fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
lh_sulc = nib.load(fsaverage5.sulc_left).agg_data()
rh_sulc = nib.load(fsaverage5.sulc_right).agg_data()

# Gray anatomical gyral/sulcal background
lh_bg = np.where(lh_sulc > 0, 0.1, 0.4)
rh_bg = np.where(rh_sulc > 0, 0.1, 0.4)

# Colormap: Sulcal gray values (0.0 to 0.5) and Red ROI overlay (1.0)
color_nodes = [0.0, 0.5, 0.51, 1.0]
colors = ["#b0b0b0", "#e0e0e0", "#d62728", "#d62728"]
cmap_mixed = LinearSegmentedColormap.from_list("Sulc_ROI", list(zip(color_nodes, colors)))

# --- 3. Discover and Pair Labels ---
label_files = glob.glob(os.path.join(LABEL_DIR, "*.label"))
roi_pairs = {}

for path in label_files:
    fname = os.path.basename(path)
    
    # Detect hemisphere from prefixes (lh., lh_, l, L) or suffixes (_lh, _l)
    if re.search(r'(^lh[\._]|_lh$|^l(?=[a-zA-Z]))', fname, re.IGNORECASE):
        hemi = 'lh'
    elif re.search(r'(^rh[\._]|_rh$|^r(?=[a-zA-Z]))', fname, re.IGNORECASE):
        hemi = 'rh'
    else:
        hemi = 'lh'
        
    # Strip extension
    base_name = re.sub(r'(\.label$)', '', fname, flags=re.IGNORECASE)
    # Strip 'lh.' or 'rh_' patterns
    base_name = re.sub(r'(^lh[\._]|^rh[\._]|_lh$|_rh$)', '', base_name, flags=re.IGNORECASE)
    # Strip leading 'l' or 'r' regardless of subsequent casing (handles ldacc, ldACC, rdmpfc, etc.)
    base_name = re.sub(r'^[lr](?=[a-zA-Z])', '', base_name, flags=re.IGNORECASE)
    
    # Store standard lowercase base name to ensure matches across variations
    key = base_name.lower()
    
    if key not in roi_pairs:
        roi_pairs[key] = {'lh': None, 'rh': None, 'display_name': base_name}
    roi_pairs[key][hemi] = path

print(f"Discovered {len(roi_pairs)} unique base ROIs to extract.")

# --- 4. Process Each Region ---
for key, info in roi_pairs.items():
    base_roi = info['display_name']
    paths = info
    print(f"\nProcessing ROI: {base_roi}")
    
    lh_mask = np.zeros(N_VERTICES, dtype=np.int32)
    rh_mask = np.zeros(N_VERTICES, dtype=np.int32)
    
    # Extract Left Hemisphere
    if paths['lh'] and os.path.exists(paths['lh']):
        lh_verts = read_label(paths['lh'])
        valid_lh = lh_verts[lh_verts < N_VERTICES]
        lh_mask[valid_lh] = 1
        print(f"  - Loaded LH: {len(valid_lh)} vertices from {os.path.basename(paths['lh'])}")
        
    # Extract Right Hemisphere
    if paths['rh'] and os.path.exists(paths['rh']):
        rh_verts = read_label(paths['rh'])
        valid_rh = rh_verts[rh_verts < N_VERTICES]
        rh_mask[valid_rh] = 1
        print(f"  - Loaded RH: {len(valid_rh)} vertices from {os.path.basename(paths['rh'])}")
        
    # Save standard binary .npy masks matching downstream GLM naming
    lh_save = os.path.join(OUTPUT_MASK_DIR, f"lh_{base_roi}_fsavg5.npy")
    rh_save = os.path.join(OUTPUT_MASK_DIR, f"rh_{base_roi}_fsavg5.npy")
    np.save(lh_save, lh_mask)
    np.save(rh_save, rh_mask)
    print(f"  - Saved masks: {lh_save} and {rh_save}")
    
    # Generate 2D surface projection with both hemispheres on one plot
    corr_L = lh_bg.copy()
    corr_R = rh_bg.copy()
    corr_L[lh_mask.astype(bool)] = 1.0
    corr_R[rh_mask.astype(bool)] = 1.0
    
    img = brain_plot(
        [corr_L, corr_R],
        cmap=cmap_mixed,
        vmin=0.0,
        vmax=1.0,
        surf_type="inflated"
    )
    
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Bilateral ROI: {base_roi}", fontsize=13, fontweight='bold')
    
    plot_file = os.path.join(PLOT_OUTPUT_DIR, f"roi_plot_{base_roi}.png")
    plt.savefig(plot_file, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)

print("\nDone! All ROI masks match the subject fsaverage5 denoised space and bilateral plots are saved.")
