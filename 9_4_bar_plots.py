import os
import re
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests
from utils import load_mask  

# --- Configuration Paths ---
ROI_MASK_DIR = "/mnt/labdata/got_project/test/ROI_analysis/9_3_freesurfer_create_roi_masks"
GLM_DATA_DIR = "/mnt/labdata/got_project/test/brain_plot_data_output/9_8_noFamSplit_glm_pain_base_convolved9_8_noFamSplit"
PARTICIPANTS_TSV = "/mnt/labdata/got_project/test/ROI_analysis/participants.tsv"
OUTPUT_DIR = os.path.join(ROI_MASK_DIR, "9_8_combined_broad_bar_graphs_output")

GROUP_GRAPHS_DIR = os.path.join(OUTPUT_DIR, "combined_broad_bar_graphs")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "roi_extracted_all_pain_types.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GROUP_GRAPHS_DIR, exist_ok=True)

# --- Define Region Categories ---
REGION_DEFINITIONS = {
    1: {"name": "Pain Regions", "rois": ["AI", "MCC", "dACC"], "color": "#d62728"},
    2: {"name": "Theory of Mind (ToM)", "rois": ["TPJ", "dmpfc", "PCC"], "color": "#1f77b4"},
    3: {"name": "Face", "rois": ["FFA", "OFA", "pSTS"], "color": "#e48416"},
    4: {"name": "Body", "rois": ["EBA", "FBA"], "color": "#2ca02c"},
    5: {"name": "Control", "rois": ["V1"], "color": "#e6e332"}
}

# 1. Discover ROIs dynamically from .npy mask directory
discovered_files = glob.glob(os.path.join(ROI_MASK_DIR, "*_fsavg5.npy"))
file_rois = set()
for f in discovered_files:
    fname = os.path.basename(f)
    match = re.match(r'^(?:lh|rh)_(.+)_fsavg5\.npy$', fname)
    if match:
        file_rois.add(match.group(1))

RAW_ROIS = sorted(list(file_rois))
print(f"Discovered {len(RAW_ROIS)} ROI masks in {ROI_MASK_DIR}: {RAW_ROIS}")

def normalize_name(s):
    s = re.sub(r'^[lr](?:h)?[-_]?', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

def match_roi_to_category(roi_col):
    norm_col = normalize_name(roi_col)
    for sec_idx, sec_info in REGION_DEFINITIONS.items():
        for target in sec_info["rois"]:
            norm_target = normalize_name(target)
            if norm_target == norm_col:
                return sec_idx
    return None

# 2. Parse Participants Metadata
sub_df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
id_col = 'participant_id' if 'participant_id' in sub_df.columns else sub_df.columns[0]
sub_df['Subject_ID'] = sub_df[id_col].apply(lambda x: f"sub-{x}" if not str(x).startswith('sub-') else str(x))

# 3. Load Medial Wall Cortical Masks
lh_cortical_mask = load_mask('hemi-L').astype(bool)
rh_cortical_mask = load_mask('hemi-R').astype(bool)

roi_cortex_masks = {}
for roi in RAW_ROIS:
    lh_path = os.path.join(ROI_MASK_DIR, f"lh_{roi}_fsavg5.npy")
    rh_path = os.path.join(ROI_MASK_DIR, f"rh_{roi}_fsavg5.npy")
    
    if os.path.exists(lh_path):
        raw_lh = np.load(lh_path).astype(bool)
        lh_m = raw_lh[lh_cortical_mask] if len(raw_lh) == len(lh_cortical_mask) else raw_lh
    else:
        lh_m = None

    if os.path.exists(rh_path):
        raw_rh = np.load(rh_path).astype(bool)
        rh_m = raw_rh[rh_cortical_mask] if len(raw_rh) == len(rh_cortical_mask) else raw_rh
    else:
        rh_m = None
    
    roi_cortex_masks[roi] = {'lh': lh_m, 'rh': rh_m}

# --- STEP 1: Extract and Build Long Data Frame ---
print("Extracting beta values from unified GLM...")
all_extracted_data = []
skipped_subs = []

for idx, row in sub_df.iterrows():
    sub = row['Subject_ID']
    lh_npz = os.path.join(GLM_DATA_DIR, f"{sub}_hemi-L.npz")
    rh_npz = os.path.join(GLM_DATA_DIR, f"{sub}_hemi-R.npz")

    if not (os.path.exists(lh_npz) and os.path.exists(rh_npz)):
        skipped_subs.append(sub)
        continue
        
    with np.load(lh_npz) as l_arr, np.load(rh_npz) as r_arr:
        lh_betas = l_arr['betas']
        rh_betas = r_arr['betas']
        
        if 'regressor_names' in l_arr:
            names = [str(n).lower() for n in l_arr['regressor_names']]
        else:
            names = ['physical', 'mental', 'vicarious']

    for pain_type in ['Physical', 'Mental', 'Vicarious']:
        p_key = pain_type.lower()
        if p_key not in names:
            continue
            
        c_idx = names.index(p_key)

        subject_results = {
            'Subject': sub,
            'Group': str(row['group']).strip(),         
            'Familiarity': str(row['familiarity']).strip(),
            'Pain_Type': pain_type
        }
        
        for roi in RAW_ROIS:
            hemi_means = []
            lh_m = roi_cortex_masks[roi]['lh']
            rh_m = roi_cortex_masks[roi]['rh']

            if lh_m is not None and lh_m.sum() > 0 and lh_m.shape[0] == lh_betas.shape[1]:
                hemi_means.append(np.nanmean(lh_betas[c_idx, :][lh_m]))

            if rh_m is not None and rh_m.sum() > 0 and rh_m.shape[0] == rh_betas.shape[1]:
                hemi_means.append(np.nanmean(rh_betas[c_idx, :][rh_m]))
            
            if len(hemi_means) > 0:
                subject_results[roi] = float(np.nanmean(hemi_means))
            else:
                subject_results[roi] = np.nan
            
        all_extracted_data.append(subject_results)

if skipped_subs:
    print(f"Notice: Skipped {len(skipped_subs)} participants missing .npz files: {skipped_subs}")

df = pd.DataFrame(all_extracted_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Aggregated tracking matrix saved to: {OUTPUT_CSV} (Total records: {len(df)})\n")

# --- STEP 2: Map Discovered ROIs into Sections ---
ordered_rois = []
section_boundaries = []
section_centers = []
current_idx = 0

available_cols = [c for c in df.columns if c not in ['Subject', 'Group', 'Familiarity', 'Pain_Type']]

for sec_idx in sorted(REGION_DEFINITIONS.keys()):
    sec_info = REGION_DEFINITIONS[sec_idx]
    sec_name = sec_info["name"]
    sec_color = sec_info["color"]
    
    sec_rois = [col for col in available_cols if match_roi_to_category(col) == sec_idx]
    
    if not sec_rois:
        print(f"Notice: No masks detected on disk for '{sec_name}'. Skipping section.")
        continue
        
    ordered_rois.extend(sec_rois)
    start_idx = current_idx
    current_idx += len(sec_rois)
    end_idx = current_idx
    
    section_boundaries.append(end_idx - 0.5)
    section_centers.append(((start_idx + end_idx - 1) / 2, sec_name, sec_color))

if section_boundaries:
    section_boundaries = section_boundaries[:-1]

print(f"Plotting ROIs: {ordered_rois}")

# --- Helper: t-Test Engine with Benjamini-Hochberg FDR ---
def compute_ttest_contrasts_fdr(plot_data_df, rois, test_pairs, is_paired=True):
    """
    Computes paired t-tests (within-subject: Physical vs Mental) or 
    two-sample independent t-tests (between-subjects: Familiar vs Unfamiliar),
    followed by Benjamini-Hochberg FDR correction across ROIs.
    """
    raw_results = []
    
    for roi in rois:
        for cond_a, cond_b in test_pairs:
            df_a = plot_data_df[plot_data_df['Condition'] == cond_a][['Subject', roi]].dropna()
            df_b = plot_data_df[plot_data_df['Condition'] == cond_b][['Subject', roi]].dropna()
            
            p_val = 1.0
            if is_paired:
                merged = pd.merge(df_a, df_b, on='Subject', suffixes=('_a', '_b'))
                if len(merged) >= 3:
                    _, p_val = ttest_rel(merged[f"{roi}_a"], merged[f"{roi}_b"])
            else:
                vals_a = df_a[roi]
                vals_b = df_b[roi]
                if len(vals_a) >= 2 and len(vals_b) >= 2:
                    _, p_val = ttest_ind(vals_a, vals_b, equal_var=False)  # Welch's t-test

            raw_results.append({'roi': roi, 'cond_a': cond_a, 'cond_b': cond_b, 'p_raw': p_val})
            
    if not raw_results:
        return {}
        
    res_df = pd.DataFrame(raw_results)
    _, p_fdr, _, _ = multipletests(res_df['p_raw'], method='fdr_bh')
    res_df['p_fdr'] = p_fdr
    
    sig_lookup = {}
    for _, r in res_df.iterrows():
        sig_lookup[(r['roi'], r['cond_a'], r['cond_b'])] = r['p_fdr']
    return sig_lookup

# --- STEP 3: Define Target Plot Configurations ---
plot_configs = [
    {
        "filename": "1_control_vicarious_vs_nopain.png",
        "title": "Control Group: Vicarious Pain vs. Implicit Baseline",
        "prepare_fn": lambda d: (
            d.query("Group == 'Control' and Pain_Type == 'Vicarious'")
             .assign(Condition="Vicarious Pain")
        ),
        "hue": "Condition",
        "test_pairs": [],
        "paired": False
    },
    {
        "filename": "2_control_physical_vs_mental.png",
        "title": "Control Group: Physical Pain vs. Mental Pain",
        "prepare_fn": lambda d: (
            d.query("Group == 'Control' and Pain_Type in ['Physical', 'Mental']")
             .assign(Condition=lambda x: x["Pain_Type"])
        ),
        "hue": "Condition",
        "test_pairs": [("Physical", "Mental")],
        "paired": True
    },
    {
        "filename": "3_dp_vicarious_vs_nopain.png",
        "title": "DP Group: Vicarious Pain vs. Implicit Baseline",
        "prepare_fn": lambda d: (
            d.query("Group == 'DP' and Pain_Type == 'Vicarious'")
             .assign(Condition="Vicarious Pain")
        ),
        "hue": "Condition",
        "test_pairs": [],
        "paired": False
    },
    {
        "filename": "4_dp_physical_vs_mental.png",
        "title": "DP Group: Physical Pain vs. Mental Pain",
        "prepare_fn": lambda d: (
            d.query("Group == 'DP' and Pain_Type in ['Physical', 'Mental']")
             .assign(Condition=lambda x: x["Pain_Type"])
        ),
        "hue": "Condition",
        "test_pairs": [("Physical", "Mental")],
        "paired": True
    },
    {
        "filename": "5_control_familiarity_pain_vs_nopain.png",
        "title": "Control Group: Familiar vs. Unfamiliar (Viewing Pain)",
        "prepare_fn": lambda d: (
            d.query("Group == 'Control' and Pain_Type == 'Vicarious'")
             .assign(Condition=lambda x: x["Familiarity"])
        ),
        "hue": "Condition",
        "test_pairs": [("Familiar", "Unfamiliar")],
        "paired": False
    },
    {
        "filename": "6_dp_familiarity_pain_vs_nopain.png",
        "title": "DP Group: Familiar vs. Unfamiliar (Viewing Pain)",
        "prepare_fn": lambda d: (
            d.query("Group == 'DP' and Pain_Type == 'Vicarious'")
             .assign(Condition=lambda x: x["Familiarity"])
        ),
        "hue": "Condition",
        "test_pairs": [("Familiar", "Unfamiliar")],
        "paired": False
    },
    {
        "filename": "7_control_familiarity_physical_vs_mental.png",
        "title": "Control Group: Familiar vs. Unfamiliar across Physical & Mental Pain",
        "prepare_fn": lambda d: (
            d.query("Group == 'Control' and Pain_Type in ['Physical', 'Mental']")
             .assign(Condition=lambda x: x["Pain_Type"] + " (" + x["Familiarity"] + ")")
        ),
        "hue": "Condition",
        "test_pairs": [
            ("Physical (Familiar)", "Physical (Unfamiliar)"),
            ("Mental (Familiar)", "Mental (Unfamiliar)")
        ],
        "paired": False
    },
    {
        "filename": "8_dp_familiarity_physical_vs_mental.png",
        "title": "DP Group: Familiar vs. Unfamiliar across Physical & Mental Pain",
        "prepare_fn": lambda d: (
            d.query("Group == 'DP' and Pain_Type in ['Physical', 'Mental']")
             .assign(Condition=lambda x: x["Pain_Type"] + " (" + x["Familiarity"] + ")")
        ),
        "hue": "Condition",
        "test_pairs": [
            ("Physical (Familiar)", "Physical (Unfamiliar)"),
            ("Mental (Familiar)", "Mental (Unfamiliar)")
        ],
        "paired": False
    }
]

# --- STEP 4: Render Bar Charts ---
for cfg in plot_configs:
    plot_df = cfg["prepare_fn"](df)
    
    if plot_df.empty or len(ordered_rois) == 0:
        print(f"Skipping {cfg['filename']} due to empty data filter.")
        continue
        
    df_long = plot_df.melt(
        id_vars=['Subject', 'Group', 'Familiarity', 'Pain_Type', 'Condition'],
        value_vars=ordered_rois,
        var_name='ROI',
        value_name='Beta_Value'
    )
    
    fig_width = max(18, len(ordered_rois) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8.5))
    sns.set_theme(style="whitegrid", context="talk")
    
    sns.barplot(
        data=df_long,
        x='ROI',
        y='Beta_Value',
        hue=cfg["hue"],
        order=ordered_rois,
        errorbar='se',          
        palette='Set2',
        capsize=0.08,
        ax=ax
    )
    
    hue_levels = list(df_long[cfg["hue"]].unique())
    bar_peaks = []
    bar_troughs = []
    
    for roi in ordered_rois:
        for cond in hue_levels:
            vals = plot_df[plot_df["Condition"] == cond][roi].dropna()
            if len(vals) > 0:
                mean_val = vals.mean()
                sem_val = vals.sem() if len(vals) > 1 else 0.0
                bar_peaks.append(mean_val + sem_val)
                bar_troughs.append(mean_val - sem_val)

    highest_bar = max(bar_peaks) if bar_peaks else 0.05
    lowest_bar = min(bar_troughs) if bar_troughs else -0.05
    
    dynamic_max = max(0.0, highest_bar)
    dynamic_min = min(0.0, lowest_bar)
    data_spread = dynamic_max - dynamic_min
    if data_spread == 0:
        data_spread = max(abs(highest_bar) * 0.5, 0.01)

    bracket_spacing = data_spread * 0.08
    current_bracket_top = dynamic_max

    # Exact bar x-coordinate lookup map
    x_positions = {}
    for patch in ax.patches:
        patch_x = patch.get_x() + patch.get_width() / 2.0
        roi_idx = int(round(patch_x))
        if 0 <= roi_idx < len(ordered_rois):
            roi_name = ordered_rois[roi_idx]
            hue_idx = int(np.round((patch_x - roi_idx) * len(hue_levels))) % len(hue_levels)
            cond_name = hue_levels[hue_idx]
            x_positions[(roi_name, cond_name)] = patch_x

    # Compute t-test contrasts with FDR correction
    fdr_results = compute_ttest_contrasts_fdr(
        plot_df, 
        ordered_rois, 
        cfg["test_pairs"], 
        is_paired=cfg.get("paired", False)
    ) if cfg["test_pairs"] else {}

    # Place Significance Brackets
    if cfg["test_pairs"]:
        for i, roi in enumerate(ordered_rois):
            for cond_a, cond_b in cfg["test_pairs"]:
                p_fdr = fdr_results.get((roi, cond_a, cond_b), 1.0)
                
                if p_fdr < 0.05:
                    vals_a = plot_df[plot_df["Condition"] == cond_a][roi].dropna()
                    vals_b = plot_df[plot_df["Condition"] == cond_b][roi].dropna()
                    
                    peak_a = vals_a.mean() + vals_a.sem() if len(vals_a) > 1 else vals_a.mean()
                    peak_b = vals_b.mean() + vals_b.sem() if len(vals_b) > 1 else vals_b.mean()
                    local_peak = max(peak_a, peak_b)
                    
                    y_bar = max(local_peak + bracket_spacing, current_bracket_top + (bracket_spacing * 0.4))
                    y_text = y_bar + (bracket_spacing * 0.15)
                    current_bracket_top = y_text + (bracket_spacing * 1.1)
                    
                    x_a = x_positions.get((roi, cond_a), i - 0.2)
                    x_b = x_positions.get((roi, cond_b), i + 0.2)
                    
                    stars = "***" if p_fdr < 0.001 else ("**" if p_fdr < 0.01 else "*")
                        
                    ax.plot(
                        [x_a, x_a, x_b, x_b], 
                        [y_bar - (bracket_spacing * 0.1), y_bar, y_bar, y_bar - (bracket_spacing * 0.1)], 
                        color="black", lw=1.2, clip_on=False
                    )
                    ax.text(
                        (x_a + x_b) / 2.0, y_text, stars, 
                        ha='center', va='bottom', color="black", fontweight="bold", fontsize=13, clip_on=False
                    )

    # Dynamic y-limits and formatting
    y_pad_bottom = dynamic_min - (data_spread * 0.10)
    y_pad_top = current_bracket_top + (data_spread * 0.22)
    ax.set_ylim(y_pad_bottom, y_pad_top)

    ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    
    if data_spread < 0.01:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    elif data_spread < 0.1:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Draw section dividers
    for b in section_boundaries:
        ax.axvline(x=b, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        
    ax.axhline(0, color='black', linewidth=1.0, linestyle='-', zorder=2)
    
    # Category Header badges
    curr_ymin, curr_ymax = ax.get_ylim()
    label_y = curr_ymax - (curr_ymax - curr_ymin) * 0.02
    for center_x, label_text, label_color in section_centers:
        ax.text(
            center_x, label_y, label_text, 
            ha='center', va='top', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=label_color, edgecolor='black', alpha=0.3),
            clip_on=False
        )
        
    ax.set_xticklabels(ordered_rois, rotation=45, ha='right')
    ax.set_ylabel(r"Neural Activation ($\beta$ Values)", labelpad=12)
    ax.set_xlabel("Functional Region of Interest (ROI)", labelpad=15)
    ax.set_title(cfg["title"], pad=35, fontweight="bold", fontsize=15)
    ax.legend(title="Condition", frameon=True, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.subplots_adjust(top=0.82, bottom=0.28, left=0.07, right=0.88)
    
    save_path = os.path.join(GROUP_GRAPHS_DIR, cfg["filename"])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

print("\nProcessing complete! Bar graphs generated successfully.")