#create high level regressor for nuisance regressors input for main regressor code

import os
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

# --- Global Timing Configuration ---
TR = 2.0                 # fMRI TR in seconds[cite: 6]
N_SCANS = 389            # Target timepoints[cite: 6]
ANNOTATION_INTERVAL = 4  # Each annotation row represents 4 seconds[cite: 6]
FRAME_TIMES = np.arange(N_SCANS) * TR #[cite: 6]
SHIFT_TRS = int(4 / TR)  # 4-second shift = 2 TRs[cite: 6]

# --- File Paths ---
input_csv = "/mnt/labdata/got_project/test/regressor_convolved/regressor code and hrf files/GOT Annotations Combined - Sheet14.csv"
output_dir = "/mnt/labdata/got_project/test/brain_plot_data_output/final_regressor_updated"
output_csv = os.path.join(output_dir, "combined_glm_regressors.csv")

def upsample_annotations(raw_series, target_trs=N_SCANS):
    """
    Expands 4s annotations to 2s TR resolution and matches target length.[cite: 6]
    """
    # Each 4-second window spans 2 TRs of 2 seconds each[cite: 6]
    repeat_factor = int(ANNOTATION_INTERVAL // TR) #[cite: 6]
    upsampled = np.repeat(raw_series, repeat_factor) #[cite: 6]
    
    if len(upsampled) >= target_trs:
        return upsampled[:target_trs] #[cite: 6]
    else:
        return np.pad(upsampled, (0, target_trs - len(upsampled)), mode="constant") #[cite: 6]

def convolve_feature(amplitudes, frame_times=FRAME_TIMES, tr=TR):
    """
    Convolves non-zero parametric amplitude weights with the SPM HRF.[cite: 6]
    """
    active_indices = np.where(amplitudes > 0)[0] #[cite: 6]
    
    if len(active_indices) == 0:
        return np.zeros(len(frame_times))
        
    onsets = active_indices * tr #[cite: 6]
    durations = [tr] * len(onsets) #[cite: 6]
    active_weights = amplitudes[active_indices] #[cite: 6]
    
    exp_condition = (onsets, durations, active_weights) #[cite: 6]
    
    convolved_signal, _ = compute_regressor(
        exp_condition=exp_condition,
        hrf_model="spm", #[cite: 6]
        frame_times=frame_times
    )
    return convolved_signal.flatten() #[cite: 6]

# --- 1. Load Data ---
df_raw = pd.read_csv(input_csv)
feature_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

regressors_dict = {}

# --- 2. Upsample and Convolve Each Feature ---
for col in feature_cols:
    raw_vals = df_raw[col].values
    
    # 2a. Upsample from 4s intervals to 2s TRs[cite: 6]
    base_amplitudes = upsample_annotations(raw_vals, target_trs=N_SCANS) #[cite: 6]
    
    # 2b. Convolve Base regressor[cite: 6]
    regressors_dict[f"{col}_base"] = convolve_feature(base_amplitudes) #[cite: 6]
    
    # 2c. Shift forward by 4 seconds (2 TRs) and convolve[cite: 6]
    shifted_amplitudes = np.zeros_like(base_amplitudes) #[cite: 6]
    shifted_amplitudes[SHIFT_TRS:] = base_amplitudes[:-SHIFT_TRS] #[cite: 6]
    regressors_dict[f"{col}_shift_4s"] = convolve_feature(shifted_amplitudes) #[cite: 6]

# --- 3. Save to Composite CSV ---
os.makedirs(output_dir, exist_ok=True) #[cite: 6]
df_combined = pd.DataFrame(regressors_dict)
df_combined.to_csv(output_csv, index=False)

print(f"Generated {df_combined.shape[0]} TRs across {df_combined.shape[1]} total regressors.")
print(f"Saved combined file to: {output_csv}")
