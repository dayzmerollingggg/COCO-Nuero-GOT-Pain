import os
import sys
import json
import numpy as np
import pandas as pd
import neuroboros as nb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from scipy.special import legendre
from joblib import Parallel, delayed, parallel_backend

# --- Path Configuration ---
SCRIPTS_DIR = '/mnt/labdata/got_project/test/brain_plot_glm_code'
sys.path.append(SCRIPTS_DIR)

from utils import load_mask 
from brainplotlib import brain_plot

LOW_LVL_DIR = '/mnt/labdata/got_project/test/low_level_data/new2sec_lowlvl' 
DNN_DIR = '/mnt/labdata/got_project/test/alexnet' 
PROJ_DIR = '/mnt/labdata/got_project' 
FMRI_DATA_DIR = os.path.join(PROJ_DIR, 'data') 
DATA_DIR = os.path.join(PROJ_DIR, 'test/brain_plot_data_output') 
TSV_PATH = os.path.join(DATA_DIR, 'participants.tsv')

# --- Global Configurations ---
HEMIS = ['hemi-L', 'hemi-R']


# --- Participant Parsing Helpers ---
def get_subjects_by_subgroup(group_name, familiarity_name=None, tsv_path=TSV_PATH):
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Could not locate participants metadata file at: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    df['group'] = df['group'].str.strip()
    
    if familiarity_name is not None:
        df['familiarity'] = df['familiarity'].str.strip()
        filtered_df = df[(df['group'] == group_name) & (df['familiarity'] == familiarity_name)]
    else:
        filtered_df = df[df['group'] == group_name]
        
    return filtered_df['participant_id'].tolist()

def get_all_tsv_subjects(tsv_path=TSV_PATH):
    df = pd.read_csv(tsv_path, sep='\t')
    return df['participant_id'].dropna().tolist()


# --- Nuisance Matrix Engineering ---
def legendre_polynomials(n_tp, poly_order=2):
    x = np.linspace(-1, 1, n_tp)
    poly = np.zeros((n_tp, poly_order))
    for i, order in enumerate(range(1, poly_order + 1)):
        poly[:, i] = legendre(order)(x)
    return poly

def get_got_nuisance(subj, include_low_level=True):
    motion_parameters = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    first_temporal_derivatives = [m+'_derivative1' for m in motion_parameters]
    columns = motion_parameters + first_temporal_derivatives

    preproc_dir = os.path.join(FMRI_DATA_DIR, 'derivatives')
    fmriprep_dir = os.path.join(preproc_dir, f'fmriprep/{subj}/func')
    confounds_fn = f'{subj}_task-GoT_desc-confounds_timeseries.tsv'
    confounds_file = os.path.join(fmriprep_dir, confounds_fn)
    raw_confounds_df = pd.read_csv(confounds_file, sep='\t')
    
    raw_confounds = np.nan_to_num(raw_confounds_df[columns].values)
    
    if include_low_level:
        dnn_dir = os.path.join(DNN_DIR)
        dnn_fn = 'visual_features_pca1_new.npy'
        dnn_data = np.load(os.path.join(dnn_dir, dnn_fn), allow_pickle=True)[:-1]
        
        low_lvl_dir = os.path.join(LOW_LVL_DIR)
        audio_data = pd.read_csv(os.path.join(low_lvl_dir, 'audio_pitch_output.csv'))[:-1]
        hsv_data = pd.read_csv(os.path.join(low_lvl_dir, 'hsv_output.csv'))[:-1]
        motion_data = pd.read_csv(os.path.join(low_lvl_dir, 'motion_energy_output.csv'))[:-1]
        
        audio_regressors = audio_data[['Average_Pitch_Hz', 'Average_Amplitude']].values
        hsv_regressors = hsv_data[['Average_H', 'Average_S', 'Average_V']].values
        motion_regressors = motion_data[['Average_Motion_Energy']].values
        
        combined_nuisance = np.concatenate(
            (dnn_data, audio_regressors, hsv_regressors, motion_regressors, raw_confounds), 
            axis=1
        )
    else:
        combined_nuisance = raw_confounds
    
    confounds = np.nan_to_num(zscore(combined_nuisance, axis=0))
    poly = legendre_polynomials(n_tp=confounds.shape[0])
    return np.concatenate((confounds, poly), axis=1) 

def get_high_level_regressors():
    regressor_fn = os.path.join(DATA_DIR, 'final_regressor_updated', 'combined_glm_regressors.csv')
    df_high = pd.read_csv(regressor_fn)
    
    # Drop the shifted faces regressor while keeping the convolved base feature
    cols_to_keep = [
        col for col in df_high.columns 
        if not (('face' in col.lower()) and ('shift' in col.lower()))
    ]
    return df_high[cols_to_keep]


# --- Core GLM Engine & Multiple Comparisons Corrections ---
def multiple_comparisons_fdr_positive(t_stats, df, raw_mask, q=0.05):
    full_sig_mask = np.zeros_like(raw_mask, dtype=float) 
    p_values = 1 - stats.t.cdf(t_stats, df)
    p_values[t_stats <= 0] = 1.0
    
    p_flat = p_values.flatten()
    m = len(p_flat)
    
    sort_indices = np.argsort(p_flat)
    p_sorted = p_flat[sort_indices]
    
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * q
    below_threshold = p_sorted <= thresholds
    
    if not np.any(below_threshold):
        return full_sig_mask
        
    max_k_idx = np.where(below_threshold)[0][-1]
    p_threshold = p_sorted[max_k_idx]
    
    significant_brain_vertices = (p_flat <= p_threshold) & (t_stats.flatten() > 0)
    full_sig_mask[raw_mask == 1] = significant_brain_vertices.astype(float)
    return full_sig_mask

def check_design_matrix_correlation(X, feature_names, subject_id, hemi):
    df_X = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_X.corr()
    
    num_features = len(feature_names)
    fig_width = max(12, int(num_features * 0.3))
    fig_height = max(10, int(num_features * 0.25))

    plt.figure(figsize=(fig_width, fig_height))
    show_annotations = True if num_features <= 20 else False
    
    sns.heatmap(
        corr_matrix, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        annot=show_annotations, 
        fmt=".2f",
        cbar=True
    )
    plt.title(f'Design Matrix Check - {subject_id} ({hemi})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(DATA_DIR, f'{subject_id}_{hemi}_design_matrix_corr.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"--> Diagnostic matrix trace updated to: {save_path}")

def run_glm_manual(subj, hemi, regressor_norm, include_low_level=True, include_high_level=True):
    mask = load_mask(hemi)
    denoised_dir = os.path.join(FMRI_DATA_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_task-GoT_space-fsaverage5_{hemi}_denoised.npy'
    
    target_brain_path = os.path.join(denoised_dir, data_fn)
    if not os.path.exists(target_brain_path):
        raise FileNotFoundError(f"Missing fMRI data array for subject: {subj} ({hemi})")
        
    data = np.load(target_brain_path)[:, mask]
    
    # 1. Prepare both Standardized (z-scored) and Unstandardized (mean-centered) signals
    ds_standardized = np.nan_to_num(zscore(data, axis=0))
    ds_unstandardized = np.nan_to_num(data - np.nanmean(data, axis=0))

    nuisance = get_got_nuisance(subj, include_low_level=include_low_level)                  
    
    task_values = regressor_norm.values
    task_names = list(regressor_norm.columns)
    n_task_regressors = task_values.shape[1]
    matrix_blocks = [task_values]
    
    if include_high_level:
        high_level_regressors = get_high_level_regressors()  
        matrix_blocks.append(high_level_regressors.values)
        
    matrix_blocks.append(nuisance)

    X = np.hstack(matrix_blocks)
    
    dead_columns = np.std(X, axis=0) == 0
    if np.any(dead_columns):
        X = X[:, ~dead_columns]
    
    if subj == get_all_tsv_subjects()[0] and hemi == 'hemi-L':
        high_lvl_labels = list(high_level_regressors.columns) if include_high_level else []
        nuisance_labels = [f'Nuisance_{i}' for i in range(X.shape[1] - n_task_regressors - len(high_lvl_labels))]
        all_feature_names = task_names + high_lvl_labels + nuisance_labels
        check_design_matrix_correlation(X, all_feature_names, subj, hemi)

    # 2. Fit GLM for both standardized and unstandardized signals
    beta_std = np.linalg.lstsq(X, ds_standardized, rcond=-1)[0]
    beta_raw = np.linalg.lstsq(X, ds_unstandardized, rcond=-1)[0]
    
    n_timepoints = ds_standardized.shape[0]
    n_predictors = X.shape[1]
    df = n_timepoints - n_predictors
    
    # Statistical tests are computed from the standardized model (scale-invariant)
    residuals = ds_standardized - np.dot(X, beta_std)
    sigma = np.sqrt(np.sum(residuals**2, axis=0) / df)
    
    cov = np.dot(X.T, X)
    inv_cov = np.linalg.inv(cov)
    
    out_betas_raw = []
    out_betas_std = []
    out_ts = []
    out_sig_masks = []
    out_names = []

    # 3. Extract Individual Regressors
    for i in range(n_task_regressors):
        se = sigma * np.sqrt(inv_cov[i, i])
        t_stat = beta_std[i, :] / se
        sig_mask = multiple_comparisons_fdr_positive(t_stat, df, raw_mask=mask, q=0.05)
        
        out_betas_raw.append(beta_raw[i, :])
        out_betas_std.append(beta_std[i, :])
        out_ts.append(t_stat)
        out_sig_masks.append(sig_mask)
        out_names.append(task_names[i])

    # 4. Built-in Composite Contrast: Vicarious (Physical + Mental)
    lower_task_names = [name.lower() for name in task_names]
    if 'physical' in lower_task_names and 'mental' in lower_task_names and 'vicarious' not in lower_task_names:
        p_idx = lower_task_names.index('physical')
        m_idx = lower_task_names.index('mental')

        c = np.zeros(X.shape[1])
        c[p_idx] = 0.5
        c[m_idx] = 0.5

        contrast_beta_std = 0.5 * beta_std[p_idx, :] + 0.5 * beta_std[m_idx, :]
        contrast_beta_raw = 0.5 * beta_raw[p_idx, :] + 0.5 * beta_raw[m_idx, :]
        
        c_var = np.dot(c, np.dot(inv_cov, c))
        contrast_se = sigma * np.sqrt(c_var)
        contrast_t = contrast_beta_std / contrast_se
        contrast_sig_mask = multiple_comparisons_fdr_positive(contrast_t, df, raw_mask=mask, q=0.05)

        out_betas_raw.append(contrast_beta_raw)
        out_betas_std.append(contrast_beta_std)
        out_ts.append(contrast_t)
        out_sig_masks.append(contrast_sig_mask)
        out_names.append('vicarious')

    return (
        np.array(out_betas_raw), 
        np.array(out_betas_std), 
        np.array(out_ts), 
        np.array(out_sig_masks), 
        out_names
    )


def parallel_pipe_safe(glm_dir, subj, hemi, regressor_norm, include_low_level, include_high_level):
    try:
        out_fn = os.path.join(glm_dir, f'{subj}_{hemi}.npz')
        betas_raw, betas_std, ts, sig_masks, reg_names = run_glm_manual(
            subj, hemi, regressor_norm, 
            include_low_level=include_low_level, 
            include_high_level=include_high_level
        )
        np.savez(
            out_fn, 
            betas=betas_raw,                        # Unstandardized betas
            standardized_betas=betas_std,           # Standardized betas
            ts=ts, 
            sig_masks=sig_masks,
            regressor_names=np.array(reg_names, dtype=str)
        )
        return subj, True, None
    except FileNotFoundError as err:
        print(f"--> [Skipping Thread] {subj} ({hemi}) | Reason: {err}")
        return subj, False, str(err)
    except Exception as unhandled_err:
        print(f"--> [Thread Error] Participant {subj} ({hemi}): {unhandled_err}")
        return subj, False, str(unhandled_err)


# --- Direct Contrast Computations ---
def run_physical_vs_mental_contrast(glm_dir, avg_data_dir, fig_dir, target_contrasts, 
                                     split_familiar, file_change, 
                                     include_low_level=True, include_high_level=True):
    lower_contrasts = [c.lower() for c in target_contrasts]
    if 'physical' not in lower_contrasts or 'mental' not in lower_contrasts:
        print("--> [Skipping Contrast] Both 'physical' and 'mental' must be present in target_contrasts.")
        return

    phys_idx = lower_contrasts.index('physical')
    ment_idx = lower_contrasts.index('mental')

    if split_familiar:
        subgroups = {
            'control_familiar':   ('Control', 'Familiar'),
            'control_unfamiliar': ('Control', 'Unfamiliar'),
            'DP_familiar':        ('DP', 'Familiar'),
            'DP_unfamiliar':      ('DP', 'Unfamiliar')
        }
    else:
        subgroups = {
            'control_combined':   ('Control', None),
            'DP_combined':        ('DP', None)
        }

    left_mask = load_mask('hemi-L')
    right_mask = load_mask('hemi-R')
    combined_raw_mask = np.concatenate([left_mask, right_mask])

    print("\n--> Running Paired Contrast: Physical vs. Mental...")

    for subgroup_label, (grp, fam) in subgroups.items():
        subjects = get_subjects_by_subgroup(grp, fam)
        diff_betas = []

        for subj in subjects:
            subj_phys, subj_ment = [], []
            has_complete_data = True

            for hemi in HEMIS:
                out_fn = os.path.join(glm_dir, f'{subj}_{hemi}.npz')
                if not os.path.exists(out_fn):
                    has_complete_data = False
                    break
                
                with np.load(out_fn) as data:
                    raw_names = data['regressor_names']
                    names = [n.decode('utf-8').lower() if isinstance(n, bytes) else str(n).lower() for n in raw_names]
                    
                    p_i = names.index('physical') if 'physical' in names else phys_idx
                    m_i = names.index('mental') if 'mental' in names else ment_idx
                    
                    betas = data['betas']
                    subj_phys.append(betas[p_i, :])
                    subj_ment.append(betas[m_i, :])

            if has_complete_data:
                b_phys = np.concatenate(subj_phys)
                b_ment = np.concatenate(subj_ment)
                diff_betas.append(b_phys - b_ment)

        if not diff_betas:
            continue

        diff_matrix = np.stack(diff_betas, axis=1)
        group_t, fdr_conserved_t = calculate_group_fdr(diff_matrix, raw_mask=combined_raw_mask)
        
        contrast_name = 'physical_vs_mental'
        np.save(os.path.join(avg_data_dir, f'{subgroup_label}_{contrast_name}_tstat.npy'), group_t)
        np.save(os.path.join(avg_data_dir, f'{subgroup_label}_{contrast_name}_fdr_mask.npy'), fdr_conserved_t)

    plot_all_contrasts_fdr_split(
        between_dir=None,
        data_dir=avg_data_dir,
        fig_dir=fig_dir,
        glm_dir=glm_dir,
        file_change=file_change,
        include_low_level=include_low_level,
        include_high_level=include_high_level,
        split_familiar=split_familiar,
        contrast_names=['physical_vs_mental']
    )


def run_mental_vs_physical_contrast(glm_dir, avg_data_dir, fig_dir, target_contrasts, 
                                     split_familiar, file_change, 
                                     include_low_level=True, include_high_level=True):
    lower_contrasts = [c.lower() for c in target_contrasts]
    if 'physical' not in lower_contrasts or 'mental' not in lower_contrasts:
        print("--> [Skipping Contrast] Both 'physical' and 'mental' must be present in target_contrasts.")
        return

    phys_idx = lower_contrasts.index('physical')
    ment_idx = lower_contrasts.index('mental')

    if split_familiar:
        subgroups = {
            'control_familiar':   ('Control', 'Familiar'),
            'control_unfamiliar': ('Control', 'Unfamiliar'),
            'DP_familiar':        ('DP', 'Familiar'),
            'DP_unfamiliar':      ('DP', 'Unfamiliar')
        }
    else:
        subgroups = {
            'control_combined':   ('Control', None),
            'DP_combined':        ('DP', None)
        }

    left_mask = load_mask('hemi-L')
    right_mask = load_mask('hemi-R')
    combined_raw_mask = np.concatenate([left_mask, right_mask])

    print("\n--> Running Paired Contrast: Mental vs. Physical...")

    for subgroup_label, (grp, fam) in subgroups.items():
        subjects = get_subjects_by_subgroup(grp, fam)
        diff_betas = []

        for subj in subjects:
            subj_phys, subj_ment = [], []
            has_complete_data = True

            for hemi in HEMIS:
                out_fn = os.path.join(glm_dir, f'{subj}_{hemi}.npz')
                if not os.path.exists(out_fn):
                    has_complete_data = False
                    break
                
                with np.load(out_fn) as data:
                    raw_names = data['regressor_names']
                    names = [n.decode('utf-8').lower() if isinstance(n, bytes) else str(n).lower() for n in raw_names]
                    
                    p_i = names.index('physical') if 'physical' in names else phys_idx
                    m_i = names.index('mental') if 'mental' in names else ment_idx
                    
                    betas = data['betas']
                    subj_phys.append(betas[p_i, :])
                    subj_ment.append(betas[m_i, :])

            if has_complete_data:
                b_phys = np.concatenate(subj_phys)
                b_ment = np.concatenate(subj_ment)
                diff_betas.append(b_ment - b_phys)

        if not diff_betas:
            continue

        diff_matrix = np.stack(diff_betas, axis=1)
        group_t, fdr_conserved_t = calculate_group_fdr(diff_matrix, raw_mask=combined_raw_mask)
        
        contrast_name = 'mental_vs_physical'
        np.save(os.path.join(avg_data_dir, f'{subgroup_label}_{contrast_name}_tstat.npy'), group_t)
        np.save(os.path.join(avg_data_dir, f'{subgroup_label}_{contrast_name}_fdr_mask.npy'), fdr_conserved_t)

    plot_all_contrasts_fdr_split(
        between_dir=None,
        data_dir=avg_data_dir,
        fig_dir=fig_dir,
        glm_dir=glm_dir,
        file_change=file_change,
        include_low_level=include_low_level,
        include_high_level=include_high_level,
        split_familiar=split_familiar,
        contrast_names=['mental_vs_physical']
    )


# --- Group Level Processing Across Dynamic Labels ---
def calculate_between_group_fdr(group1_betas, group2_betas, raw_mask, q=0.05):
    t_stats, p_vals = stats.ttest_ind(group1_betas, group2_betas, axis=1, equal_var=True)
    df_between = group1_betas.shape[1] + group2_betas.shape[1] - 2
    
    group_mask = multiple_comparisons_fdr_positive(t_stats, df_between, raw_mask=raw_mask, q=q)
    full_t_stats = np.zeros_like(raw_mask, dtype=float)
    full_t_stats[raw_mask == 1] = t_stats
    
    conserved_t_map = full_t_stats * group_mask
    conserved_t_map[group_mask == 0] = np.nan
    return full_t_stats, conserved_t_map

def calculate_group_fdr(group_betas, raw_mask, q=0.05):
    n_subjects = group_betas.shape[1]
    df_group = n_subjects - 1
    
    group_mean = np.mean(group_betas, axis=1)
    group_std = np.std(group_betas, axis=1, ddof=1)
    group_sem = group_std / np.sqrt(n_subjects)
    group_t = group_mean / group_sem
    
    group_mask = multiple_comparisons_fdr_positive(group_t, df_group, raw_mask=raw_mask, q=q)
    full_group_t = np.zeros_like(raw_mask, dtype=float)
    full_group_t[raw_mask == 1] = group_t
    
    conserved_t_map = full_group_t * group_mask
    conserved_t_map[group_mask == 0] = np.nan
    return full_group_t, conserved_t_map

def average_glm_fdr_dynamic(between_data_dir, avg_data_dir, glm_dir, contrast_names, split_familiar, include_low_level=True, include_high_level=True):
    if split_familiar:
        subgroups = {
            'control_familiar':   ('Control', 'Familiar'),
            'control_unfamiliar': ('Control', 'Unfamiliar'),
            'DP_familiar':        ('DP', 'Familiar'),
            'DP_unfamiliar':      ('DP', 'Unfamiliar')
        }
    else:
        subgroups = {
            'control_combined':   ('Control', None),
            'DP_combined':        ('DP', None)
        }
    
    left_mask = load_mask('hemi-L')
    right_mask = load_mask('hemi-R')
    combined_raw_mask = np.concatenate([left_mask, right_mask])
    
    for contrast_index, contrast in enumerate(contrast_names):
        group_matrices = {}
        for subgroup_label, (grp, fam) in subgroups.items():
            subjects = get_subjects_by_subgroup(grp, fam)
            all_betas = []
            
            for subj in subjects:
                subj_data = []
                has_complete_data = True
                
                for hemi in HEMIS:
                    out_fn = os.path.join(glm_dir, f'{subj}_{hemi}.npz')
                    if not os.path.exists(out_fn):
                        has_complete_data = False
                        break
                    with np.load(out_fn) as d:
                        raw_names = d['regressor_names']
                        names = [n.decode('utf-8').lower() if isinstance(n, bytes) else str(n).lower() for n in raw_names]
                        target_c = contrast.lower()
                        
                        if target_c in names:
                            c_idx = names.index(target_c)
                        else:
                            c_idx = contrast_index
                            
                        data = d['betas'][c_idx, :]
                    subj_data.append(data)
                
                if has_complete_data:
                    all_betas.append(np.concatenate(subj_data)) 

            if not all_betas:
                print(f"--> [Warning] No completed subject data found for subgroup: {subgroup_label}. Skipping matrix step.")
                continue

            group_beta_matrix = np.stack(all_betas, axis=1)
            group_matrices[subgroup_label] = group_beta_matrix
            
            group_t, fdr_conserved_t = calculate_group_fdr(group_beta_matrix, raw_mask=combined_raw_mask)
            np.save(os.path.join(avg_data_dir, f'{subgroup_label}_{contrast}_tstat.npy'), group_t)
            np.save(os.path.join(avg_data_dir, f'{subgroup_label}_{contrast}_fdr_mask.npy'), fdr_conserved_t)
            
        if split_familiar and 'control_familiar' in group_matrices and 'DP_familiar' in group_matrices:
            t_between, fdr_conserved_between = calculate_between_group_fdr(
                group_matrices['control_familiar'], 
                group_matrices['DP_familiar'], 
                raw_mask=combined_raw_mask
            )
            np.save(os.path.join(between_data_dir, f'controlFam_vs_DPFam_{contrast}_tstat.npy'), t_between)
            np.save(os.path.join(between_data_dir, f'controlFam_vs_DPFam_{contrast}_fdr_mask.npy'), fdr_conserved_between)
        elif not split_familiar and 'control_combined' in group_matrices and 'DP_combined' in group_matrices:
            t_between, fdr_conserved_between = calculate_between_group_fdr(
                group_matrices['control_combined'], 
                group_matrices['DP_combined'], 
                raw_mask=combined_raw_mask
            )
            np.save(os.path.join(between_data_dir, f'control_vs_DP_combined_{contrast}_tstat.npy'), t_between)
            np.save(os.path.join(between_data_dir, f'control_vs_DP_combined_{contrast}_fdr_mask.npy'), fdr_conserved_between)


# --- Visualization / Plotting Grid Matrix ---
def plot_brains_grid_split(plot_rows, col_titles, row_titles, fig_title, vmax=3, cbar_label='t'):
    vmin = -1 * vmax
    num_rows = len(plot_rows)
    num_cols = len(col_titles)
    
    fig_width = 5 * num_cols + 2
    fig = plt.figure(figsize=(fig_width, 9))
    
    row_y_starts = [0.52, 0.16]
    row_height = 0.32
    
    for r in range(num_rows):
        y_start = row_y_starts[r]
        for c in range(num_cols):
            x_start = 0.08 + (c * (0.85 / num_cols)) 
            x_width = 0.80 / num_cols
            
            ax = fig.add_axes([x_start, y_start, x_width, row_height])
            img = brain_plot(plot_rows[r][c], vmin=vmin, vmax=vmax, cmap='seismic')
            ax.imshow(img)
            ax.axis('off')
            
            if r == 0:
                ax.text(0.5, 1.05, col_titles[c], transform=ax.transAxes,
                        fontsize=16, va='bottom', ha='center', fontweight='bold')
                
    fig.suptitle(fig_title, fontsize=22, y=0.96, fontweight='bold')

    fig.text(0.01, 0.68, row_titles[0], rotation=90, va='center', ha='left', fontsize=16, fontweight='bold')
    fig.text(0.01, 0.32, row_titles[1], rotation=90, va='center', ha='left', fontsize=16, fontweight='bold')

    cbar_ax = fig.add_axes([0.35, 0.06, 0.30, 0.02]) 
    norm = plt.Normalize(vmin, vmax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap='seismic'),
        cax=cbar_ax, 
        orientation='horizontal',
        label=f'{cbar_label}-value'
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.label.set_fontsize(13)
    return fig


def plot_all_contrasts_fdr_split(between_dir, data_dir, fig_dir, glm_dir, file_change, include_low_level, include_high_level, split_familiar, contrast_names):
    row_labels = ['Raw T-Stat', 'FDR Thresholded (Pos Only)']
    low_status = "+Low-Level" if include_low_level else "-Low-Level"
    high_status = "+High-Level" if include_high_level else "-High-Level"
    title_metadata = f"({low_status}, {high_status})"

    target_groups = ['control', 'DP']

    for contrast in contrast_names:
        for group in target_groups:
            if split_familiar:
                sub_conditions = [f'{group}_familiar', f'{group}_unfamiliar']
                col_headers = [f'{group.upper()} Familiar', f'{group.upper()} Unfamiliar']
                filename_suffix = "split"
            else:
                sub_conditions = [f'{group}_combined']
                col_headers = [f'{group.upper()} Combined']
                filename_suffix = "combined"

            t_row = []
            mask_row = []
            max_t = 0
            
            for cond in sub_conditions:
                t_path = os.path.join(data_dir, f'{cond}_{contrast}_tstat.npy')
                mask_path = os.path.join(data_dir, f'{cond}_{contrast}_fdr_mask.npy')
                
                if not os.path.exists(t_path) or not os.path.exists(mask_path):
                    continue
                    
                t_data = np.load(t_path)
                t_row.append(t_data)
                
                fdr_conserved_data = np.load(mask_path)
                mask_row.append(fdr_conserved_data)
                
                max_t = max(max_t, np.percentile(np.abs(t_data), 98)) 

            if not t_row:
                print(f"--> [Warning] Missing files for contrast {contrast}, Group: {group}")
                continue

            plot_data = [t_row, mask_row]
            fig_title = f"{group.upper()} Group - {contrast.capitalize()} {title_metadata}"
            
            fig = plot_brains_grid_split(
                plot_rows=plot_data, 
                col_titles=col_headers, 
                row_titles=row_labels, 
                fig_title=fig_title,
                vmax=max_t, 
                cbar_label='t'
            )
            
            save_fn = os.path.join(fig_dir, f'{group}_{contrast}_{file_change}_{filename_suffix}.png')
            fig.savefig(save_fn, bbox_inches='tight')
            plt.close(fig)


# --- Core Pipeline Orchestration ---
def run_regressors(base_type='pain', shift_type='base', regressor_type='convolved', 
                   additional_data='', include_low_level=True, include_high_level=True, 
                   split_familiar=True, use_combined_pain=False):
    
    regressors_dir = os.path.join(DATA_DIR, 'regressor_pain_updated_389tr')
    
    if use_combined_pain:
        target_contrasts = ['vicarious'] 
        filename_base = 'pain_base_convolved_vicarious_only.csv' 
        feature_suffix = "_combinedPain"
    else:
        target_contrasts = ['physical', 'mental']
        filename_base = 'pain_base_convolved.csv'
        feature_suffix = ""
        
    if not include_low_level: feature_suffix += "_noLowLevel"
    if not include_high_level: feature_suffix += "_noHighLevel"
    if not split_familiar: feature_suffix += "_noFamSplit"
    
    file_change = f'{base_type}_{shift_type}_{regressor_type}{additional_data}{feature_suffix}'
    
    GLM_DIR = os.path.join(DATA_DIR, f'{additional_data}{feature_suffix}_glm_{file_change}')
    AVERAGED_DATA_DIR = os.path.join(DATA_DIR, f'{additional_data}{feature_suffix}_averaged_data_{file_change}')
    BETWEEN_DATA_DIR = os.path.join(DATA_DIR, f'{additional_data}{feature_suffix}_averaged_data{file_change}')
    FIG_DIR = os.path.join(DATA_DIR, f'{additional_data}{feature_suffix}_figures_{file_change}')
    
    os.makedirs(AVERAGED_DATA_DIR, exist_ok=True)
    os.makedirs(GLM_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(BETWEEN_DATA_DIR, exist_ok=True)
    
    norm_regressors_fn = os.path.join(regressors_dir, filename_base)
    regressor_norm = pd.read_csv(norm_regressors_fn)
    
    print(f"Loaded regressor target map. Shape: {regressor_norm.shape} Columns: {list(regressor_norm.columns)}")
    sliced_regressors = regressor_norm[target_contrasts]
    
    subjects = get_all_tsv_subjects()
    print(f"Loaded {len(subjects)} total targets out of tracked participant records.")
    
    jobs = []
    for subj in subjects:
        for hemi in HEMIS:
            jobs.append(delayed(parallel_pipe_safe)(
                GLM_DIR, subj, hemi, sliced_regressors, include_low_level, include_high_level
            ))
            
    print(f"--> Spawning 4 active multi-processing workers across {len(jobs)} computation pipelines...")
    with parallel_backend("loky", inner_max_num_threads=1):
        worker_results = Parallel(n_jobs=4, verbose=1)(jobs)

    skipped_subjects = {}
    for subj, success, err in worker_results:
        if not success:
            skipped_subjects[subj] = err

    if skipped_subjects:
        skip_log_path = os.path.join(GLM_DIR, "skipped_participants.log")
        with open(skip_log_path, 'w') as f:
            for s, err in skipped_subjects.items():
                f.write(f"{s}: {err}\n")
        print(f"--> [Warning] {len(skipped_subjects)} subject pipelines failed or had missing data. Logged to: {skip_log_path}")

    final_output_contrasts = list(target_contrasts)
    if 'physical' in [c.lower() for c in target_contrasts] and 'mental' in [c.lower() for c in target_contrasts]:
        if 'vicarious' not in [c.lower() for c in final_output_contrasts]:
            final_output_contrasts.append('vicarious')

    manifest_data = {
        'glm_dir': GLM_DIR,
        'regressors_fitted': list(sliced_regressors.columns),
        'saved_contrasts': final_output_contrasts,
        'total_tsv_subjects': len(subjects),
        'successful_subjects': len(subjects) - len(skipped_subjects),
        'include_low_level': include_low_level,
        'include_high_level': include_high_level,
        'split_familiar': split_familiar,
        'use_combined_pain': use_combined_pain
    }
    manifest_path = os.path.join(GLM_DIR, 'glm_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=4)
    print(f"--> Saved GLM execution manifest to: {manifest_path}")

    print("\nIndividual estimation passes finished. Progressing to group rendering matrices...")
    
    average_glm_fdr_dynamic(BETWEEN_DATA_DIR, AVERAGED_DATA_DIR, GLM_DIR, final_output_contrasts, split_familiar)
    
    plot_all_contrasts_fdr_split(
        BETWEEN_DATA_DIR, AVERAGED_DATA_DIR, FIG_DIR, GLM_DIR, 
        file_change, include_low_level, include_high_level, split_familiar, final_output_contrasts
    )

    if not use_combined_pain:
        run_physical_vs_mental_contrast(
            glm_dir=GLM_DIR,
            avg_data_dir=AVERAGED_DATA_DIR,
            fig_dir=FIG_DIR,
            target_contrasts=target_contrasts,
            split_familiar=split_familiar,
            file_change=file_change,
            include_low_level=include_low_level,
            include_high_level=include_high_level
        )

        run_mental_vs_physical_contrast(
            glm_dir=GLM_DIR,
            avg_data_dir=AVERAGED_DATA_DIR,
            fig_dir=FIG_DIR,
            target_contrasts=target_contrasts,
            split_familiar=split_familiar,
            file_change=file_change,
            include_low_level=include_low_level,
            include_high_level=include_high_level
        )

    return GLM_DIR


if __name__ == "__main__":
    active_glm_dir = run_regressors(
        base_type='pain', 
        shift_type='base', 
        regressor_type='convolved', 
        additional_data='9_8_new',
        include_low_level=True,
        include_high_level=True,
        split_familiar=False,
        use_combined_pain=False
    )