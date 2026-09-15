import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import compute_regressor

# --- Global Configuration ---
TR = 2.0                 # fMRI TR in seconds
N_SCANS = 389            # Total number of fMRI volumes
FRAME_TIMES = np.arange(N_SCANS) * TR

# --- File Paths ---
data_dir = "/mnt/labdata/got_project/test/brain_plot_data_output/regressor_pain_updated_389tr"
input_csv = os.path.join(data_dir, "pain_base_raw_389tr.csv")
output_csv = os.path.join(data_dir, "pain_convolved_389tr.csv")
output_plot = os.path.join(data_dir, "pain_raw_vs_convolved_overlay.png")

# --- 1. Load Raw Data ---
df_raw = pd.read_csv(input_csv)
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# --- 2. Convolve Each Feature Using the 5_9 Event-Weighting Method ---
convolved_dict = {}

for col in numeric_cols:
    amplitudes = df_raw[col].values

    # Extract non-zero events and treat each active TR as an event with amplitude weighting
    active_indices = np.where(amplitudes > 0)[0]
    onsets = active_indices * TR
    durations = [TR] * len(onsets)
    active_weights = amplitudes[active_indices]

    exp_condition = (onsets, durations, active_weights)

    # Compute the convolved regressor exactly as specified
    convolved_signal, _ = compute_regressor(
        exp_condition=exp_condition,
        hrf_model='spm',
        frame_times=FRAME_TIMES
    )
    
    convolved_dict[f"{col}_convolved"] = convolved_signal.flatten()

# --- 3. Save Convolved CSV ---
df_conv = pd.DataFrame(convolved_dict)
os.makedirs(data_dir, exist_ok=True)
df_conv.to_csv(output_csv, index=False)
print(f"Saved convolved regressors ({df_conv.shape}) to {output_csv}")

# --- 4. Plotting (Combined from 9_7_plot.py) ---
conditions = ["physical", "mental", "vicarious"]
matched_pairs = []

# Match based on common keywords, fallback to 1-to-1 index matching
for cond in conditions:
    raw_c = next((c for c in df_raw.columns if cond in c.lower()), None)
    conv_c = next((c for c in df_conv.columns if cond in c.lower()), None)
    if raw_c and conv_c:
        matched_pairs.append((cond.capitalize(), raw_c, conv_c))

if not matched_pairs:
    min_len = min(len(df_raw.columns), len(df_conv.columns))
    for i in range(min_len):
        matched_pairs.append((f"Feature {i+1}", df_raw.columns[i], df_conv.columns[i]))

n_plots = len(matched_pairs)
fig, axes = plt.subplots(
    n_plots, 1, figsize=(16, 3.5 * n_plots), sharex=True, squeeze=False
)

for i, (title, raw_c, conv_c) in enumerate(matched_pairs):
    ax1 = axes[i, 0]

    # Raw ratings on primary axis
    line1 = ax1.step(
        df_raw.index,
        df_raw[raw_c].values,
        where="post",
        color="#7f7f7f",
        alpha=0.6,
        linewidth=1.2,
        label=f"Raw ({raw_c})",
    )
    ax1.fill_between(
        df_raw.index,
        df_raw[raw_c].values,
        step="post",
        color="#7f7f7f",
        alpha=0.15,
    )
    ax1.set_ylabel("Raw Rating (Unconvolved)", color="#4f4f4f", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#4f4f4f")
    ax1.set_ylim(-0.1, max(1.1, df_raw[raw_c].max() * 1.15))

    # Convolved signal on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        df_conv.index,
        df_conv[conv_c].values,
        color="#d62728",
        linewidth=2.0,
        label=f"Convolved ({conv_c})",
    )
    ax2.set_ylabel("Convolved Amplitude (HRF)", color="#d62728", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax1.set_title(f"Overlay: {title}", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

axes[-1, 0].set_xlabel(f"TR Index (0 to {N_SCANS - 1})", fontsize=12)
plt.tight_layout()

plt.savefig(output_plot, dpi=180, bbox_inches="tight")
print(f"Saved comparison figure to: {output_plot}")
plt.show()