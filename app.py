# ─────────────────────────────────────────
# app.py
# EEGNet+XAI Alzheimer's Demo
# Streamlit Dashboard
# ─────────────────────────────────────────

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="EEG Alzheimer's Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
ALL_CHANNELS = [
    'Fp1','Fp2','F7','F3','Fz',
    'F4','F8','T3','C3','Cz',
    'C4','T4','T5','P3','Pz',
    'P4','T6','O1','O2'
]

XAI_CHANNELS = [
    'T5','O1','P4','Pz','O2',
    'Fz','C4','P3','C3','T3'
]

XAI_IMPORTANCE = {
    'T5':0.937,'O1':0.918,
    'P4':0.812,'Pz':0.805,
    'O2':0.762,'Fz':0.750,
    'C4':0.655,'P3':0.582,
    'C3':0.553,'T3':0.546
}

XAI_IDX = [ALL_CHANNELS.index(ch)
           for ch in XAI_CHANNELS]

THRESHOLD = 0.45
FS        = 250
EP_LEN    = 500

CH_COORDS = {
    'Fp1':(-0.3, 0.9),'Fp2':(0.3, 0.9),
    'F7': (-0.7, 0.5),'F3': (-0.4, 0.5),
    'Fz': (0.0,  0.5),'F4': (0.4,  0.5),
    'F8': (0.7,  0.5),'T3': (-0.9, 0.0),
    'C3': (-0.5, 0.0),'Cz': (0.0,  0.0),
    'C4': (0.5,  0.0),'T4': (0.9,  0.0),
    'T5': (-0.7,-0.5),'P3': (-0.4,-0.5),
    'Pz': (0.0, -0.5),'P4': (0.4, -0.5),
    'T6': (0.7, -0.5),'O1': (-0.3,-0.9),
    'O2': (0.3, -0.9)
}

# ─────────────────────────────────────────
# KAGGLE DOWNLOAD
# ─────────────────────────────────────────
@st.cache_resource
def download_from_kaggle():
    try:
        dl_dir = os.path.join(
            os.path.expanduser('~'),
            'eeg_checkpoints')
        os.makedirs(dl_dir, exist_ok=True)

        # If already downloaded, skip
        check_file = os.path.join(
            dl_dir,
            'p2_xai_loso_ch10.pkl')
        if os.path.exists(check_file):
            return True, "✅ Files ready"

        # Try Streamlit secrets first
        # (Streamlit Cloud deployment)
        try:
            import json
            kaggle_dir = os.path.expanduser(
                '~/.kaggle')
            os.makedirs(
                kaggle_dir, exist_ok=True)
            creds = {
                "username": st.secrets[
                    "kaggle_username"],
                "key": st.secrets[
                    "kaggle_key"]
            }
            kaggle_json = os.path.join(
                kaggle_dir, 'kaggle.json')
            with open(
                    kaggle_json, 'w') as f:
                json.dump(creds, f)
            os.chmod(kaggle_json, 0o600)
        except Exception:
            # Local: use existing
            # ~/.kaggle/kaggle.json
            pass

        # Verify kaggle.json exists
        kaggle_json = os.path.expanduser(
            '~/.kaggle/kaggle.json')
        if not os.path.exists(kaggle_json):
            return (False,
                    "kaggle.json not found "
                    "at ~/.kaggle/kaggle.json")

        # Download only required files
        import subprocess
        files_needed = [
            'p2_xai_loso_ch10.pkl',
            'p2_xai_simple.pkl',
            'p2_loso_results_final.pkl',
            'p2_freq_ablation.pkl',
            'p2_final_results.pkl',
        ]

        for fname in files_needed:
            fpath = os.path.join(
                dl_dir, fname)
            if os.path.exists(fpath):
                continue
            cmd = [
                'kaggle',
                'datasets', 'download',
                'siddhu2021/'
                'eeg-ad-phase2-checkpoints',
                '--file', fname,
                '-p', dl_dir,
                '--unzip'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True)
            if result.returncode != 0:
                return (
                    False,
                    f"Failed {fname}: "
                    f"{result.stderr}")

        return True, "✅ Files loaded"

    except Exception as e:
        return False, f"Error: {e}"

# ─────────────────────────────────────────
# LOAD PKL FILES
# No splits.pkl — too large (2GB)
# All subject info from xai_loso_ch10
# ─────────────────────────────────────────
@st.cache_resource
def load_all_files():
    dl_dir = os.path.join(
        os.path.expanduser('~'),
        'eeg_checkpoints')

    def load_pkl(fname):
        path = os.path.join(dl_dir, fname)
        with open(path, 'rb') as f:
            return pickle.load(f)

    try:
        model  = load_pkl(
            'p2_xai_loso_ch10.pkl')
        xai    = load_pkl(
            'p2_xai_simple.pkl')
        loso   = load_pkl(
            'p2_loso_results_final.pkl')
        freq   = load_pkl(
            'p2_freq_ablation.pkl')
        final  = load_pkl(
            'p2_final_results.pkl')
        return (model, xai, loso,
                freq, final, True)
    except Exception as e:
        st.error(f"Load error: {e}")
        return (None,)*5 + (False,)

# ─────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────
def plot_topo_map():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    head = plt.Circle(
        (0, 0), 1.0,
        fill=False,
        color='black',
        linewidth=2.5)
    ax.add_patch(head)

    ax.plot([0, 0], [1.0, 1.15],
            'k-', linewidth=2)

    for sign in [-1, 1]:
        ax.plot(
            [sign*1.0, sign*1.1],
            [0.1, 0.0],
            'k-', linewidth=2)
        ax.plot(
            [sign*1.0, sign*1.1],
            [-0.1, 0.0],
            'k-', linewidth=2)

    for ch, (x, y) in CH_COORDS.items():
        if ch not in XAI_IMPORTANCE:
            ax.plot(x, y, 'o',
                    color='lightgrey',
                    markersize=10,
                    zorder=2)
            ax.text(x, y+0.1, ch,
                    ha='center',
                    fontsize=7,
                    color='grey')

    sc = None
    for ch, imp in XAI_IMPORTANCE.items():
        x, y = CH_COORDS[ch]
        size = 300 + imp * 700
        sc   = ax.scatter(
            x, y, s=size,
            c=[imp],
            cmap='RdYlGn_r',
            vmin=0.5, vmax=1.0,
            zorder=3,
            edgecolors='black',
            linewidths=1.5)
        ax.text(x, y, ch,
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                color='white',
                zorder=4)

    if sc:
        cbar = plt.colorbar(
            sc, ax=ax,
            shrink=0.55, pad=0.05)
        cbar.set_label(
            'XAI Importance\n'
            '(SHAP + Occlusion)',
            fontsize=9)

    ax.set_title(
        'Topographic Channel Importance\n'
        'Posterior / Parieto-Temporal '
        'Dominance',
        fontsize=11,
        fontweight='bold')
    plt.tight_layout()
    return fig


def plot_xai_bar(xai_data):
    # Use paper values directly
    # matches Table in paper exactly
    imp_vals = np.array([
        XAI_IMPORTANCE[ch]
        for ch in XAI_CHANNELS])

    colors = [
        '#d62728' if v > 0.8
        else '#ff7f0e' if v > 0.6
        else '#1f77b4'
        for v in imp_vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.barh(
        XAI_CHANNELS, imp_vals,
        color=colors, alpha=0.85,
        edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, imp_vals):
        ax.text(
            val + 0.005,
            bar.get_y() +
            bar.get_height()/2,
            f'{val:.3f}',
            va='center', fontsize=9)

    ax.set_xlabel(
        'Combined XAI Importance\n'
        '(SHAP + Occlusion)',
        fontsize=10)
    ax.set_title(
        'Top-10 Channel Rankings\n'
        'Combined SHAP + Occlusion',
        fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    return fig


def plot_freq_ablation(freq_data):
    bands = ['Beta','Alpha',
             'Theta','Delta','Gamma']
    band_labels = [
        'Beta\n13-30Hz',
        'Alpha\n8-13Hz',
        'Theta\n4-8Hz',
        'Delta\n1-4Hz',
        'Gamma\n30-40Hz']

    p2_vals = [
        freq_data['ablation_results'][b][
            'delta_acc']
        for b in bands]
    p1_vals = [
        freq_data['p1_results'][b]['delta']
        for b in bands]

    x     = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - width/2, p1_vals, width,
           label='Phase 1: Severe AD (n=42)',
           color='#1f77b4', alpha=0.85,
           edgecolor='black', linewidth=0.8)

    ax.bar(x + width/2, p2_vals, width,
           label='Phase 2: All severity (n=65)',
           color='#ff7f0e', alpha=0.85,
           edgecolor='black', linewidth=0.8)

    theta_idx = bands.index('Theta')
    ax.bar(x[theta_idx] + width/2,
           p2_vals[theta_idx], width,
           color='#d62728', alpha=1.0,
           edgecolor='black', linewidth=1.5)

    ax.annotate(
        f'KEY FINDING\n'
        f'Theta: USELESS→CRITICAL\n'
        f'ΔAcc = '
        f'+{p2_vals[theta_idx]:.3f}',
        xy=(x[theta_idx] + width/2,
            p2_vals[theta_idx]),
        xytext=(
            x[theta_idx] + width/2 + 0.8,
            p2_vals[theta_idx] + 0.04),
        fontsize=9, fontweight='bold',
        color='#d62728',
        arrowprops=dict(
            arrowstyle='->',
            color='#d62728', lw=1.5))

    ax.axhline(0, color='black',
               linewidth=0.8,
               linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(
        band_labels, fontsize=10)
    ax.set_ylabel(
        'ΔAcc when band masked\n'
        '(positive = band is important)',
        fontsize=10)
    ax.set_title(
        'Severity-Dependent Biomarker Shift\n'
        'Beta dominates severe AD | '
        'Theta emerges in mixed severity',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.1, 0.75)
    plt.tight_layout()
    return fig


def plot_mmse_scatter(
        model_ckpt,
        highlight_subj=None):
    """
    Uses p2_xai_loso_ch10.pkl
    ['xai_loso'] — confirmed AUC=0.927
    All CN subjects have MMSE=30,
    jitter applied so all 29 are visible.
    """
    items = model_ckpt['xai_loso']

    mmse_ad_sev, prob_ad_sev = [], []
    mmse_ad_mod, prob_ad_mod = [], []
    mmse_cn,     prob_cn     = [], []
    cn_items                 = []

    for item in items:
        prob = item['prob']
        mmse = item['mmse']
        true = item['true']
        if true == 0:
            mmse_cn.append(mmse)
            prob_cn.append(prob)
            cn_items.append(item)
        elif mmse < 18:
            mmse_ad_sev.append(mmse)
            prob_ad_sev.append(prob)
        else:
            mmse_ad_mod.append(mmse)
            prob_ad_mod.append(prob)

    # Jitter CN x-values so all 29
    # don't stack at MMSE=30
    np.random.seed(42)
    jitter = np.random.uniform(
        -0.4, 0.4, len(mmse_cn))
    mmse_cn_jittered = [
        m + j for m, j
        in zip(mmse_cn, jitter)]

    fig, ax = plt.subplots(
        figsize=(10, 5))

    ax.scatter(
        mmse_ad_sev, prob_ad_sev,
        c='#d62728', marker='o',
        s=80, zorder=3,
        label=f'Severe AD (MMSE<18, '
              f'n={len(mmse_ad_sev)})',
        edgecolors='black',
        linewidths=0.8)

    ax.scatter(
        mmse_ad_mod, prob_ad_mod,
        c='#ff7f0e', marker='s',
        s=80, zorder=3,
        label=f'Moderate AD (MMSE 18-23, '
              f'n={len(mmse_ad_mod)})',
        edgecolors='black',
        linewidths=0.8)

    ax.scatter(
        mmse_cn_jittered, prob_cn,
        c='#1f77b4', marker='^',
        s=80, zorder=3,
        label=f'CN (n={len(mmse_cn)})',
        edgecolors='black',
        linewidths=0.8)

    # Highlight selected subject
    if highlight_subj:
        for i, item in enumerate(items):
            if item['subj'] == highlight_subj:
                # Use jittered x for CN
                if item['true'] == 0:
                    cn_idx = cn_items.index(
                        item)
                    x_val = (
                        item['mmse'] +
                        jitter[cn_idx])
                else:
                    x_val = item['mmse']

                ax.scatter(
                    x_val,
                    item['prob'],
                    c='yellow',
                    marker='*',
                    s=400,
                    zorder=5,
                    edgecolors='black',
                    linewidths=1.5,
                    label=f'Selected: '
                          f'{highlight_subj}')
                break

    ax.axhline(
        THRESHOLD,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'Threshold τ={THRESHOLD}')

    ax.set_xlabel(
        'MMSE Score', fontsize=11)
    ax.set_ylabel(
        'Predicted AD Probability',
        fontsize=11)
    ax.set_title(
        'Subject-Level Predictions '
        'vs MMSE Score\n'
        'EEGNet+XAI (10ch) — '
        'All 65 Subjects (AUC=0.927)',
        fontsize=12,
        fontweight='bold')
    ax.legend(
        fontsize=9,
        loc='upper right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 33)

    # Mark CN region
    ax.axvspan(
        28, 33,
        alpha=0.05,
        color='#1f77b4',
        zorder=0)
    ax.text(
        30.5, 1.02,
        'CN Region\n(MMSE=30)',
        ha='center',
        fontsize=8,
        color='#1f77b4')

    plt.tight_layout()
    return fig


def plot_model_auc():
    models = [
        'PSD+SVM (19ch)',
        'DeepConvNet (19ch)',
        'ShallowCN (19ch)',
        'EEGNetSE+XAI (10ch)',
        'EEGNet+ShallowCN\n'
        '+XAI cons.(10ch)',
        'EEGNet+ShallowCN\n'
        '+XAI (10ch)',
        'EEGNet+ShallowCN (19ch)',
        'EEGNet (19ch)',
        'EEGNetSE (19ch)',
        'EEGNet+XAI (10ch) ★'
    ]
    aucs = [
        0.875, 0.873, 0.878,
        0.880, 0.891, 0.910,
        0.906, 0.914, 0.919, 0.927
    ]

    colors = [
        '#d62728' if '★' in m
        else '#7f7f7f' if 'Deep' in m
        else '#1f77b4'
        for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        models, aucs,
        color=colors, alpha=0.85,
        edgecolor='black', linewidth=0.8)

    for bar, auc in zip(bars, aucs):
        ax.text(
            auc + 0.001,
            bar.get_y() +
            bar.get_height()/2,
            f'{auc:.3f}',
            va='center', fontsize=10)

    ax.axvline(
        0.927, color='#d62728',
        linestyle='--', linewidth=1.5,
        alpha=0.7,
        label='Best AUC = 0.927')
    ax.set_xlim(0.82, 0.95)
    ax.set_xlabel(
        'AUC (ROC Curve)', fontsize=11)
    ax.set_title(
        'All Phase 2 Models — AUC Comparison\n'
        'EEGNet+XAI (10ch): Best AUC with '
        'fewest parameters (1,890)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 About")
    st.markdown("""
    **EEGNet+XAI (10ch)**
    Severity-Aware EEG Classification
    of Alzheimer's Disease

    **Dataset:** OpenNeuro ds004504
    **Subjects:** 65 (29 CN, 36 AD)
    **Validation:** LOSO-CV
    **Best AUC:** 0.927
    **Parameters:** 1,890
    **Channels:** 10 (XAI-selected)
    **Threshold:** τ = 0.45
    """)

    st.markdown("---")
    st.markdown("## 📊 Key Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC", "0.927")
        st.metric("Sensitivity", "88.9%")
        st.metric("Sev. AD Sens.", "92.3%")
    with col2:
        st.metric("Accuracy", "90.8%")
        st.metric("Specificity", "93.1%")
        st.metric("Mod. AD Sens.", "87.0%")

    st.markdown("---")
    st.markdown("""
    **Authors:**
    K. Prabhakar
    Malla Siddharth Reddy
    Kaleru Akhila
    Prasad Chetti

    **CBIT(A), Hyderabad**
    """)

# ─────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────
st.title(
    "🧠 Severity-Aware EEG "
    "Alzheimer's Classification")
st.markdown(
    "**EEGNet+XAI (10ch)** | "
    "Phase 2: n=65 | AUC=0.927 | "
    "Leakage-Proof LOSO-CV")
st.markdown("---")

with st.spinner(
        "Loading model and data..."):
    ok, msg = download_from_kaggle()

if not ok:
    st.error(msg)
    st.info(
        "Add kaggle_username and "
        "kaggle_key to Streamlit Secrets")
    st.stop()

(model_ckpt, xai_data,
 loso_data, freq_data,
 final_data, loaded) = load_all_files()

if not loaded:
    st.error("Failed to load pkl files")
    st.stop()

st.success("✅ Model and data loaded")

# ─────────────────────────────────────────
# BUILD SUBJECT LOOKUP
# From p2_xai_loso_ch10.pkl['xai_loso']
# Confirmed: {fold,subj,true,prob,mmse}
# AUC=0.927 verified
# ─────────────────────────────────────────
subj_info = {}
for item in model_ckpt['xai_loso']:
    subj_info[item['subj']] = {
        'prob': item['prob'],
        'true': item['true'],
        'mmse': item['mmse']
    }

# Sort by true label then MMSE
# so AD subjects appear first
# giving better demo experience
all_subjects = sorted(
    subj_info.keys(),
    key=lambda s: (
        subj_info[s]['true'],
        subj_info[s]['mmse']))

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Patient Diagnosis",
    "🗺️  XAI Explanation",
    "📊 Biomarker Analysis",
    "📈 Model Performance"
])

# ═════════════════════════════════════════
# TAB 1 — PATIENT DIAGNOSIS
# ═════════════════════════════════════════
with tab1:
    st.header("Live Patient Diagnosis")
    st.markdown(
        "Select any subject from the "
        "65-subject cohort. The model uses "
        "**pre-computed LOSO-CV "
        "probabilities** — the exact "
        "values reported in the paper "
        "(AUC = 0.927).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Select Subject")

        options = []
        for sid in all_subjects:
            info_s = subj_info[sid]
            label  = (
                "AD" if info_s['true'] == 1
                else "CN")
            mmse   = info_s['mmse']
            options.append(
                f"{sid} | {label} | "
                f"MMSE={mmse}")

        chosen = st.selectbox(
            "Subject (all 65, "
            "sorted by severity):",
            options=options)

        subject_id = chosen.split(' | ')[0]

        info      = subj_info[subject_id]
        true_lab  = info['true']
        mmse_val  = info['mmse']
        prob_val  = info['prob']
        label_str = (
            "Alzheimer's Disease"
            if true_lab == 1
            else "Cognitively Normal")

        if mmse_val < 18:
            sev = "Severe AD"
        elif true_lab == 1:
            sev = "Moderate AD"
        else:
            sev = "Cognitively Normal"

        st.markdown("---")
        st.markdown(
            f"**Subject:** {subject_id}")
        st.markdown(
            f"**MMSE Score:** {mmse_val}")
        st.markdown(
            f"**Severity:** {sev}")
        st.markdown(
            f"**True Label:** {label_str}")

        run_btn = st.button(
            "🔍 Run Diagnosis",
            type="primary",
            use_container_width=True)

    with col2:
        st.subheader("Diagnosis Result")

        if run_btn:
            with st.spinner(
                    "Analysing EEG..."):
                st.session_state[
                    'result'] = {
                    'subject_id': subject_id,
                    'prob':       prob_val,
                    'true_lab':   true_lab,
                    'label_str':  label_str,
                    'mmse':       mmse_val,
                    'sev':        sev
                }

        if 'result' in st.session_state:
            r    = st.session_state['result']
            p    = r['prob']
            pred = (
                "Alzheimer's Disease"
                if p >= THRESHOLD
                else "Cognitively Normal")
            correct = (
                (p >= THRESHOLD
                 and r['true_lab'] == 1)
                or
                (p < THRESHOLD
                 and r['true_lab'] == 0))

            if pred == "Alzheimer's Disease":
                st.error(f"## 🔴 {pred}")
            else:
                st.success(f"## 🟢 {pred}")

            st.metric(
                "Predicted AD Probability",
                f"{p:.4f}",
                delta=f"Threshold τ = "
                      f"{THRESHOLD}")
            st.progress(float(p))

            conf = (p if p >= 0.5
                    else 1-p) * 100
            st.metric(
                "Model Confidence",
                f"{conf:.1f}%")

            st.markdown("---")
            st.markdown(
                f"**Ground Truth:** "
                f"{r['label_str']}")
            st.markdown(
                f"**MMSE Score:** "
                f"{r['mmse']}")
            st.markdown(
                f"**Severity:** "
                f"{r['sev']}")
            st.markdown(
                f"**Correct:** "
                f"{'✅ YES' if correct else '❌ NO'}")

            st.info(
                "⚠️ Research tool only. "
                "Not a clinical device.")
        else:
            st.markdown(
                "*Select a subject and "
                "click Run Diagnosis*")

    # MMSE scatter below
    if 'result' in st.session_state:
        st.markdown("---")
        st.subheader(
            "Subject Position — "
            "MMSE vs Predicted Probability")
        st.markdown(
            "⭐ marks the selected subject | "
            "All 65 subjects shown | "
            "CN subjects jittered around "
            "MMSE=30 for visibility")

        fig_scatter = plot_mmse_scatter(
            model_ckpt,
            highlight_subj=st.session_state[
                'result']['subject_id'])
        st.pyplot(fig_scatter)
        plt.close()

# ═════════════════════════════════════════
# TAB 2 — XAI EXPLANATION
# ═════════════════════════════════════════
with tab2:
    st.header(
        "XAI — Why Did the Model "
        "Decide This?")
    st.markdown(
        "Channel importance identified "
        "by combining **SHAP** "
        "(SHapley Additive exPlanations) "
        "+ **Channel Occlusion Sensitivity**.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Topographic Head Map")
        fig_topo = plot_topo_map()
        st.pyplot(fig_topo)
        plt.close()

    with col2:
        st.subheader(
            "Channel Importance Scores")
        fig_bar = plot_xai_bar(xai_data)
        st.pyplot(fig_bar)
        plt.close()

    st.markdown("---")
    st.info("""
    **Key Finding: Posterior Dominance**

    T5, O1, O2, P3, P4, Pz =
    **6 of 10** selected channels are
    posterior / parieto-temporal.

    This matches established AD
    neurophysiology: posterior cortical
    atrophy and parieto-temporal network
    disruption are hallmark features of
    Alzheimer's Disease.

    **Phase 1 vs Phase 2 contrast:**
    Phase 1 (severe AD only) highlighted
    frontal-temporal electrodes (F4, T3, F7).
    Phase 2 (all severity) shifts to
    posterior dominance — confirming that
    channel relevance is severity-dependent.
    """)

    # SHAP bar chart
    st.markdown("---")
    st.subheader(
        "SHAP Values — AD vs CN Attribution")
    st.markdown(
        "Mean |SHAP| across all 65 subjects "
        "for each of the 19 channels. "
        "Green shading = XAI-selected top-10.")

    shap_ad = xai_data['shap_ch_ad']
    shap_cn = xai_data['shap_ch_cn']

    fig_shap, ax = plt.subplots(
        figsize=(11, 4))
    x = np.arange(19)
    w = 0.35

    ax.bar(x - w/2, shap_ad, w,
           label='AD subjects',
           color='#d62728', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, shap_cn, w,
           label='CN subjects',
           color='#1f77b4', alpha=0.8,
           edgecolor='black', linewidth=0.5)

    for idx_ch in XAI_IDX:
        ax.axvspan(
            idx_ch - 0.5, idx_ch + 0.5,
            alpha=0.1, color='green',
            zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ALL_CHANNELS,
        rotation=45, fontsize=8)
    ax.set_ylabel(
        'Mean |SHAP|', fontsize=10)
    ax.set_title(
        'SHAP Channel Attribution\n'
        'Green shading = XAI-selected '
        'top-10 channels',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

# ═════════════════════════════════════════
# TAB 3 — BIOMARKER ANALYSIS
# ═════════════════════════════════════════
with tab3:
    st.header(
        "Severity-Dependent "
        "Biomarker Shift")
    st.markdown(
        "EEG frequency band importance "
        "**changes with AD severity**. "
        "Insights from severe-only cohorts "
        "cannot be transferred to "
        "mixed-severity cohorts.")

    fig_freq = plot_freq_ablation(freq_data)
    st.pyplot(fig_freq)
    plt.close()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    beta_p1  = freq_data[
        'p1_results']['Beta']['delta']
    beta_p2  = freq_data[
        'ablation_results']['Beta'][
        'delta_acc']
    theta_p1 = freq_data[
        'p1_results']['Theta']['delta']
    theta_p2 = freq_data[
        'ablation_results']['Theta'][
        'delta_acc']

    with col1:
        st.error(f"""
        **Beta Band (13-30 Hz)**

        Phase 1: CRITICAL
        ΔAcc = +{beta_p1:.3f}

        Phase 2: Important
        ΔAcc = +{beta_p2:.3f}

        Dominant in **severe AD** due to
        profound cortical disruption and
        beta desynchronisation.
        """)

    with col2:
        st.success(f"""
        **Theta Band (4-8 Hz) ⭐**

        Phase 1: USELESS
        ΔAcc = +{theta_p1:.3f}

        Phase 2: CRITICAL
        ΔAcc = +{theta_p2:.3f}

        Classic theta-slowing signature
        emerges when **moderate AD**
        subjects enter the cohort.
        """)

    with col3:
        st.warning("""
        **Clinical Implication**

        Models trained on severe-only
        cohorts learn beta features.

        Models for mixed-severity must
        capture theta slowing.

        **Frequency importance must be
        validated separately for each
        target severity population.**
        """)

# ═════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════
with tab4:
    st.header("Model Performance")
    st.markdown(
        "All 10 Phase 2 configurations "
        "under leakage-proof LOSO-CV "
        "(n=65 subjects).")

    fig_auc = plot_model_auc()
    st.pyplot(fig_auc)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EEGNet+XAI (10ch) ★")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | AUC | **0.927** |
        | Accuracy | 90.8% |
        | Sensitivity | 88.9% |
        | Specificity | 93.1% |
        | Parameters | 1,890 |
        | Channels | 10 |
        | Threshold τ | 0.45 |
        """)

    with col2:
        st.subheader("Severity-Stratified")
        df = pd.DataFrame({
            'Model': [
                'EEGNet+XAI (10ch) ★',
                'EEGNetSE (19ch)',
                'EEGNet (19ch)',
                'ShallowCN (19ch)',
                'PSD+SVM (19ch)'
            ],
            'Severe Sens.': [
                '0.923','0.846',
                '0.846','0.769','0.769'
            ],
            'Mod. Sens.': [
                '0.870','0.783',
                '0.783','0.739','0.696'
            ],
            'CN Spec.': [
                '0.931','0.966',
                '1.000','0.966','0.966'
            ]
        })
        st.dataframe(
            df, hide_index=True,
            use_container_width=True)

    st.markdown("---")
    st.subheader(
        "Novel Finding: "
        "Attention-Pruning Incompatibility")

    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **EEGNet + XAI (10ch)**

        19ch → 10ch: **+0.013 AUC**

        No internal attention.
        Pruning removes uninformative
        channels → model focuses on
        coherent posterior pattern
        → better generalisation.
        """)
    with col2:
        st.error("""
        **EEGNetSE + XAI (10ch)**

        19ch → 10ch: **−0.038 AUC**

        SE weights calibrated for
        19-channel context. Removing
        9 channels shifts feature
        distribution → attention
        amplifies wrong channels
        → performance degrades.
        """)

    st.info("""
    **Clinical Implication:**
    For electrode reduction (mobile
    deployment, emergency monitoring,
    patient comfort): use
    **EEGNet+XAI (10ch)**.
    EEGNetSE requires all 19 channels.
    """)

    st.markdown("---")
    st.subheader(
        "DeepConvNet: "
        "Catastrophic Overfitting")
    st.error("""
    DeepConvNet (154,802 parameters)
    classified ALL 29 CN subjects as AD.
    Specificity = 0.000.

    82× more parameters than
    EEGNet+XAI (10ch) → WORSE performance.

    Confirms: lightweight architectures
    are essential for small EEG cohorts.
    """)