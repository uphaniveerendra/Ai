import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, io
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score, f1_score)
from sklearn.impute import SimpleImputer
import joblib

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {"fraud":"#E74C3C","legit":"#2ECC71","primary":"#2C3E50",
          "accent":"#3498DB","warn":"#F39C12"}

# ════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE FILE IDs  ← PASTE YOUR IDs HERE
# ════════════════════════════════════════════════════════════════════
DRIVE_IDS = {
    "train/Train_Labels.csv":       "PASTE_FILE_ID_HERE",
    "train/Train_Beneficiary.csv":  "PASTE_FILE_ID_HERE",
    "train/Train_Inpatient.csv":    "PASTE_FILE_ID_HERE",
    "train/Train_Outpatient.csv":   "PASTE_FILE_ID_HERE",
    "unseen/Unseen_Labels.csv":     "PASTE_FILE_ID_HERE",
    "unseen/Unseen_Beneficiary.csv":"PASTE_FILE_ID_HERE",
    "unseen/Unseen_Inpatient.csv":  "PASTE_FILE_ID_HERE",
    "unseen/Unseen_Outpatient.csv": "PASTE_FILE_ID_HERE",
}

DATA_CACHE = {}   # in-memory store so we only download once per session

def gdrive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

@st.cache_data(show_spinner=False)
def load_csv_from_drive(key, file_id):
    import requests
    url = gdrive_url(file_id)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def load_all_data():
    """Returns (train_labels, train_bene, train_ip, train_op,
                unseen_labels, unseen_bene, unseen_ip, unseen_op)"""
    ids_configured = all(v != "PASTE_FILE_ID_HERE" for v in DRIVE_IDS.values())

    if ids_configured:
        with st.spinner("📥 Downloading data from Google Drive…"):
            tl  = load_csv_from_drive("tl",  DRIVE_IDS["train/Train_Labels.csv"])
            tb  = load_csv_from_drive("tb",  DRIVE_IDS["train/Train_Beneficiary.csv"])
            tip = load_csv_from_drive("tip", DRIVE_IDS["train/Train_Inpatient.csv"])
            top = load_csv_from_drive("top", DRIVE_IDS["train/Train_Outpatient.csv"])
            ul  = load_csv_from_drive("ul",  DRIVE_IDS["unseen/Unseen_Labels.csv"])
            ub  = load_csv_from_drive("ub",  DRIVE_IDS["unseen/Unseen_Beneficiary.csv"])
            uip = load_csv_from_drive("uip", DRIVE_IDS["unseen/Unseen_Inpatient.csv"])
            uop = load_csv_from_drive("uop", DRIVE_IDS["unseen/Unseen_Outpatient.csv"])
        return tl, tb, tip, top, ul, ub, uip, uop
    else:
        return None   # triggers upload UI

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🏥 Healthcare Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📋 Project Overview",
    "📊 Exploratory Data Analysis",
    "⚙️  Feature Engineering",
    "🤖 Model Training & Evaluation",
    "🔍 Predict on Unseen Data",
    "💡 Business Recommendations",
])

# ── Manual upload fallback (shown when Drive IDs not set) ────────────
def manual_upload_ui():
    st.warning("⚠️ Google Drive IDs not configured yet. Upload CSVs manually below, or paste the IDs into `app.py`.")
    st.subheader("📂 Upload your 8 CSV files")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training data**")
        tl_f  = st.file_uploader("Train_Labels.csv",      type="csv", key="tl")
        tb_f  = st.file_uploader("Train_Beneficiary.csv", type="csv", key="tb")
        tip_f = st.file_uploader("Train_Inpatient.csv",   type="csv", key="tip")
        top_f = st.file_uploader("Train_Outpatient.csv",  type="csv", key="top")
    with c2:
        st.markdown("**Unseen / Test data**")
        ul_f  = st.file_uploader("Unseen_Labels.csv",      type="csv", key="ul")
        ub_f  = st.file_uploader("Unseen_Beneficiary.csv", type="csv", key="ub")
        uip_f = st.file_uploader("Unseen_Inpatient.csv",   type="csv", key="uip")
        uop_f = st.file_uploader("Unseen_Outpatient.csv",  type="csv", key="uop")

    all_uploaded = all([tl_f, tb_f, tip_f, top_f, ul_f, ub_f, uip_f, uop_f])
    if all_uploaded:
        return (pd.read_csv(tl_f), pd.read_csv(tb_f),
                pd.read_csv(tip_f), pd.read_csv(top_f),
                pd.read_csv(ul_f),  pd.read_csv(ub_f),
                pd.read_csv(uip_f), pd.read_csv(uop_f))
    return None

# ── Feature engineering ──────────────────────────────────────────────
FEATURE_COLS = [
    'IP_TotalClaims','IP_TotalReimbursed','IP_AvgReimbursed','IP_MaxReimbursed',
    'IP_TotalDeductible','IP_AvgAdmitDuration','IP_AvgClaimDuration',
    'IP_UniquePatients','IP_UniquePhysicians','IP_AvgDiagCodes','IP_AvgProcCodes',
    'IP_HasOperatingPhy','OP_TotalClaims','OP_TotalReimbursed','OP_AvgReimbursed',
    'OP_MaxReimbursed','OP_TotalDeductible','OP_AvgClaimDuration',
    'OP_UniquePatients','OP_UniquePhysicians','OP_AvgDiagCodes','OP_AvgProcCodes',
    'B_AvgAge','B_DeadPatients','B_AvgChronic','B_AvgIPReimbursement',
    'B_AvgOPReimbursement','TotalClaims','TotalReimbursed',
    'IPvsOP_ClaimRatio','ReimbursedPerClaim',
]

@st.cache_data(show_spinner=False)
def build_features(_labels, _bene, _inpatient, _outpatient):
    bene = _bene.copy()
    bene['DOB'] = pd.to_datetime(bene['DOB'], errors='coerce')
    bene['DOD'] = pd.to_datetime(bene['DOD'], errors='coerce')
    bene['Age'] = (pd.Timestamp('2009-12-31') - bene['DOB']).dt.days // 365
    bene['IsDead'] = bene['DOD'].notna().astype(int)
    chronic_cols = [c for c in bene.columns if 'ChronicCond' in c]
    bene['TotalChronicConditions'] = bene[chronic_cols].apply(lambda r:(r==1).sum(), axis=1)

    ip = _inpatient.copy()
    for col in ['AdmissionDt','DischargeDt','ClaimStartDt','ClaimEndDt']:
        ip[col] = pd.to_datetime(ip[col], errors='coerce')
    ip['AdmitDuration']   = (ip['DischargeDt'] - ip['AdmissionDt']).dt.days.clip(lower=0)
    ip['ClaimDuration']   = (ip['ClaimEndDt']  - ip['ClaimStartDt']).dt.days.clip(lower=0)
    ip['DiagCodesCount']  = ip[[f'ClmDiagnosisCode_{i}' for i in range(1,11)]].notna().sum(axis=1)
    ip['ProcCodesCount']  = ip[[f'ClmProcedureCode_{i}' for i in range(1,7)]].notna().sum(axis=1)
    ip['HasOperatingPhy'] = ip['OperatingPhysician'].notna().astype(int)

    ip_agg = ip.groupby('Provider').agg(
        IP_TotalClaims      =('ClaimID','count'),
        IP_TotalReimbursed  =('InscClaimAmtReimbursed','sum'),
        IP_AvgReimbursed    =('InscClaimAmtReimbursed','mean'),
        IP_MaxReimbursed    =('InscClaimAmtReimbursed','max'),
        IP_TotalDeductible  =('DeductibleAmtPaid','sum'),
        IP_AvgAdmitDuration =('AdmitDuration','mean'),
        IP_AvgClaimDuration =('ClaimDuration','mean'),
        IP_UniquePatients   =('BeneID','nunique'),
        IP_UniquePhysicians =('AttendingPhysician','nunique'),
        IP_AvgDiagCodes     =('DiagCodesCount','mean'),
        IP_AvgProcCodes     =('ProcCodesCount','mean'),
        IP_HasOperatingPhy  =('HasOperatingPhy','sum'),
    ).reset_index()

    op = _outpatient.copy()
    for col in ['ClaimStartDt','ClaimEndDt']:
        op[col] = pd.to_datetime(op[col], errors='coerce')
    op['ClaimDuration']  = (op['ClaimEndDt'] - op['ClaimStartDt']).dt.days.clip(lower=0)
    op['DiagCodesCount'] = op[[f'ClmDiagnosisCode_{i}' for i in range(1,11)]].notna().sum(axis=1)
    op['ProcCodesCount'] = op[[f'ClmProcedureCode_{i}' for i in range(1,7)]].notna().sum(axis=1)

    op_agg = op.groupby('Provider').agg(
        OP_TotalClaims      =('ClaimID','count'),
        OP_TotalReimbursed  =('InscClaimAmtReimbursed','sum'),
        OP_AvgReimbursed    =('InscClaimAmtReimbursed','mean'),
        OP_MaxReimbursed    =('InscClaimAmtReimbursed','max'),
        OP_TotalDeductible  =('DeductibleAmtPaid','sum'),
        OP_AvgClaimDuration =('ClaimDuration','mean'),
        OP_UniquePatients   =('BeneID','nunique'),
        OP_UniquePhysicians =('AttendingPhysician','nunique'),
        OP_AvgDiagCodes     =('DiagCodesCount','mean'),
        OP_AvgProcCodes     =('ProcCodesCount','mean'),
    ).reset_index()

    all_claims = pd.concat([ip[['Provider','BeneID']], op[['Provider','BeneID']]], ignore_index=True).drop_duplicates()
    bp = all_claims.merge(bene, on='BeneID', how='left')
    bene_agg = bp.groupby('Provider').agg(
        B_AvgAge             =('Age','mean'),
        B_DeadPatients       =('IsDead','sum'),
        B_AvgChronic         =('TotalChronicConditions','mean'),
        B_AvgIPReimbursement =('IPAnnualReimbursementAmt','mean'),
        B_AvgOPReimbursement =('OPAnnualReimbursementAmt','mean'),
    ).reset_index()

    df = _labels.copy()
    df = df.merge(ip_agg,   on='Provider', how='left')
    df = df.merge(op_agg,   on='Provider', how='left')
    df = df.merge(bene_agg, on='Provider', how='left')
    df['TotalClaims']        = df['IP_TotalClaims'].fillna(0)  + df['OP_TotalClaims'].fillna(0)
    df['TotalReimbursed']    = df['IP_TotalReimbursed'].fillna(0) + df['OP_TotalReimbursed'].fillna(0)
    df['IPvsOP_ClaimRatio']  = df['IP_TotalClaims'].fillna(0) / (df['OP_TotalClaims'].fillna(0)+1)
    df['ReimbursedPerClaim'] = df['TotalReimbursed'] / (df['TotalClaims']+1)
    if 'PotentialFraud' in df.columns:
        df['Label'] = (df['PotentialFraud']=='Yes').astype(int)
    return df

def prepare_X(df):
    X = df[FEATURE_COLS].copy()
    imp = SimpleImputer(strategy='median')
    return pd.DataFrame(imp.fit_transform(X), columns=FEATURE_COLS), imp

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ════════════════════════════════════════════════════════════════════
#  DATA LOADING — try Drive first, else manual upload
# ════════════════════════════════════════════════════════════════════
data = load_all_data()
if data is None:
    data = manual_upload_ui()

if data is None:
    st.info("👆 Please upload all 8 CSV files above to continue.")
    st.stop()

tl, tb, tip, top, ul, ub, uip, uop = data
st.sidebar.success("✅ Data loaded")

# ════════════════════════════════════════════════════════════════════
#  PAGE 1 — Project Overview
# ════════════════════════════════════════════════════════════════════
if page == "📋 Project Overview":
    st.title("🏥 Healthcare Provider Fraud Detection")
    st.markdown("### Predicting Fraudulent Insurance Providers Using Machine Learning")

    c1, c2, c3, c4 = st.columns(4)
    fraud_pct = (tl['PotentialFraud']=='Yes').mean()*100
    c1.metric("Total Providers",  f"{tl.shape[0]:,}")
    c2.metric("Fraud Rate",        f"{fraud_pct:.1f}%")
    c3.metric("Inpatient Claims",  f"{tip.shape[0]:,}")
    c4.metric("Outpatient Claims", f"{top.shape[0]:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Problem Statement")
        st.info("""
Provider fraud is one of the biggest problems in healthcare insurance.
This app builds a **binary classifier** to flag fraudulent providers.

**Common fraud types:**
- Billing for services never provided
- Duplicate claim submissions
- Upcoding — billing more expensive procedures
- Charging for covered codes on non-covered services
        """)
    with col2:
        st.subheader("📂 Dataset Summary")
        st.markdown("""
| Dataset | Rows | Description |
|---|---|---|
| Train Labels | 5,410 | Provider + Fraud label |
| Beneficiary | 138,556 | Patient KYC & chronic conditions |
| Inpatient | ~40K | Hospital admission claims |
| Outpatient | ~500K | Outpatient visit claims |
        """)

    st.markdown("---")
    st.subheader("🗺️ End-to-End Workflow")
    steps = ["Load Data","EDA","Feature Engineering","Model Training","Evaluation","Predict","Recommendations"]
    cols = st.columns(len(steps))
    for i,(col,step) in enumerate(zip(cols,steps)):
        col.markdown(f"**Step {i+1}**"); col.success(step)

# ════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    t1,t2,t3,t4 = st.tabs(["Fraud Labels","Beneficiary","Inpatient","Outpatient"])

    with t1:
        st.subheader("Fraud Label Distribution")
        fraud_counts = tl['PotentialFraud'].value_counts()
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        axes[0].bar(fraud_counts.index, fraud_counts.values,
                    color=[COLORS['fraud'],COLORS['legit']])
        axes[0].set_title("Provider Count by Label")
        for i,v in enumerate(fraud_counts.values):
            axes[0].text(i, v+20, str(v), ha='center', fontweight='bold')
        axes[1].pie(fraud_counts.values, labels=fraud_counts.index,
                    colors=[COLORS['fraud'],COLORS['legit']],
                    autopct='%1.1f%%', startangle=90)
        axes[1].set_title("Proportion")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.warning(f"⚠️ Class imbalance: {fraud_counts['Yes']} fraud vs {fraud_counts['No']} legitimate providers.")

    with t2:
        bene_c = tb.copy()
        bene_c['DOB'] = pd.to_datetime(bene_c['DOB'], errors='coerce')
        bene_c['Age'] = (pd.Timestamp('2009-12-31') - bene_c['DOB']).dt.days // 365
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(bene_c['Age'].dropna(), bins=30, color=COLORS['accent'], edgecolor='white')
            ax.set_title("Age Distribution"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
            st.pyplot(fig); plt.close()
        with col2:
            chronic_cols = [c for c in bene_c.columns if 'ChronicCond' in c]
            rates = (bene_c[chronic_cols]==1).mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6,4))
            rates.plot(kind='bar', ax=ax, color=COLORS['warn'])
            ax.set_title("Chronic Condition Prevalence")
            ax.set_xticklabels([c.replace('ChronicCond_','') for c in rates.index], rotation=45, ha='right')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total IP Claims",    f"{tip.shape[0]:,}")
        col2.metric("Avg Reimbursement",  f"${pd.to_numeric(tip['InscClaimAmtReimbursed'],errors='coerce').mean():,.0f}")
        col3.metric("Unique Providers",   f"{tip['Provider'].nunique():,}")
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        pd.to_numeric(tip['InscClaimAmtReimbursed'],errors='coerce').clip(
            upper=pd.to_numeric(tip['InscClaimAmtReimbursed'],errors='coerce').quantile(0.99)
        ).hist(bins=50, ax=axes[0], color=COLORS['accent'])
        axes[0].set_title("Reimbursement Distribution"); axes[0].set_xlabel("Amount ($)")
        tip.groupby('Provider')['ClaimID'].count().clip(upper=tip.groupby('Provider')['ClaimID'].count().quantile(0.99)
        ).hist(bins=40, ax=axes[1], color=COLORS['primary'])
        axes[1].set_title("Claims per Provider"); axes[1].set_xlabel("# Claims")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t4:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total OP Claims",    f"{top.shape[0]:,}")
        col2.metric("Avg Reimbursement",  f"${pd.to_numeric(top['InscClaimAmtReimbursed'],errors='coerce').mean():,.0f}")
        col3.metric("Unique Providers",   f"{top['Provider'].nunique():,}")
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        pd.to_numeric(top['InscClaimAmtReimbursed'],errors='coerce').clip(
            upper=pd.to_numeric(top['InscClaimAmtReimbursed'],errors='coerce').quantile(0.99)
        ).hist(bins=50, ax=axes[0], color=COLORS['legit'])
        axes[0].set_title("Reimbursement Distribution"); axes[0].set_xlabel("Amount ($)")
        top.groupby('Provider')['ClaimID'].count().clip(upper=top.groupby('Provider')['ClaimID'].count().quantile(0.99)
        ).hist(bins=40, ax=axes[1], color=COLORS['warn'])
        axes[1].set_title("Claims per Provider"); axes[1].set_xlabel("# Claims")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
#  PAGE 3 — Feature Engineering
# ════════════════════════════════════════════════════════════════════
elif page == "⚙️  Feature Engineering":
    st.title("⚙️ Feature Engineering")
    with st.spinner("Building features…"):
        df = build_features(tl, tb, tip, top)
    st.success(f"✅ {df.shape[0]} providers × {len(FEATURE_COLS)} features built")

    st.subheader("Feature Preview")
    st.dataframe(df[['Provider']+FEATURE_COLS[:10]].head(10), use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr,dtype=bool)),
                annot=False, cmap='coolwarm', center=0, ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Avg Feature Values: Fraud vs Legit")
    summary_cols = ['TotalClaims','TotalReimbursed','IP_AvgAdmitDuration',
                    'IP_UniquePhysicians','B_AvgChronic','ReimbursedPerClaim']
    stat_df = df.groupby('PotentialFraud')[summary_cols].mean().T
    stat_df.columns = ['Legitimate','Fraudulent']
    stat_df['Ratio Fraud/Legit'] = (stat_df['Fraudulent']/stat_df['Legitimate']).round(2)
    st.dataframe(stat_df.style.background_gradient(cmap='RdYlGn_r', subset=['Ratio Fraud/Legit']),
                 use_container_width=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE 4 — Model Training
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")
    with st.spinner("Building features…"):
        df = build_features(tl, tb, tip, top)
    X_raw, y = df[FEATURE_COLS], df['Label']

    st.subheader("Configure Model")
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Algorithm",["Random Forest","Gradient Boosting","Logistic Regression"])
    with col2:
        n_est = st.slider("n_estimators", 50, 300, 150, 50)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight='balanced', n_jobs=-1)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_est, random_state=42, learning_rate=0.1, max_depth=4)
    else:
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training…"):
            X_imp, imp = prepare_X(df)
            model.fit(X_imp, y)
            joblib.dump((model, imp), MODEL_PATH)

        st.success("✅ Model trained and saved!")
        y_pred  = model.predict(X_imp)
        y_proba = model.predict_proba(X_imp)[:,1]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",     f"{accuracy_score(y,y_pred):.3f}")
        c2.metric("ROC-AUC",      f"{roc_auc_score(y,y_proba):.3f}")
        c3.metric("F1-Score",     f"{f1_score(y,y_pred):.3f}")
        c4.metric("Fraud Recall", f"{classification_report(y,y_pred,output_dict=True)['1']['recall']:.3f}")

        st.subheader("5-Fold Cross Validation (ROC-AUC)")
        cv_scores = cross_val_score(model, X_imp, y, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring='roc_auc')
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(range(1,6), cv_scores, color=COLORS['accent'])
        ax.axhline(cv_scores.mean(), color=COLORS['fraud'], linestyle='--', label=f"Mean={cv_scores.mean():.3f}")
        ax.set_ylim(0,1); ax.set_xlabel("Fold"); ax.set_ylabel("ROC-AUC"); ax.legend()
        st.pyplot(fig); plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
            ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
            st.pyplot(fig); plt.close()
        with col2:
            st.subheader("ROC Curve")
            fpr,tpr,_ = roc_curve(y, y_proba)
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(fpr,tpr,color=COLORS['fraud'],label=f"AUC={roc_auc_score(y,y_proba):.3f}")
            ax.plot([0,1],[0,1],'k--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
            st.pyplot(fig); plt.close()

        if hasattr(model,'feature_importances_'):
            st.subheader("Top 15 Feature Importances")
            fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)[:15]
            fig, ax = plt.subplots(figsize=(9,4))
            fi.plot(kind='bar', ax=ax, color=COLORS['primary'])
            plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.subheader("Classification Report")
        rep = classification_report(y, y_pred, target_names=['Legit','Fraud'], output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
    else:
        st.info("👆 Click **Train Model** to begin.")

# ════════════════════════════════════════════════════════════════════
#  PAGE 5 — Predict Unseen
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict on Unseen Data":
    st.title("🔍 Predict on Unseen Data")

    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ No trained model found. Go to **Model Training** and train first.")
        st.stop()

    model, imp = joblib.load(MODEL_PATH)
    st.success("✅ Trained model loaded.")

    unseen_labels_df = ul[['Provider']].copy()
    if 'PotentialFraud' in ul.columns:
        unseen_labels_df['PotentialFraud'] = ul['PotentialFraud']

    with st.spinner("Engineering features for unseen data…"):
        df_u = build_features(unseen_labels_df, ub, uip, uop)

    X_u = df_u[FEATURE_COLS]
    X_u_imp = pd.DataFrame(imp.transform(X_u), columns=FEATURE_COLS)

    y_pred_u  = model.predict(X_u_imp)
    y_proba_u = model.predict_proba(X_u_imp)[:,1]

    results = pd.DataFrame({
        'Provider':       df_u['Provider'].values,
        'Probability':    y_proba_u.round(4),
        'PredictedClass': ['Yes' if p==1 else 'No' for p in y_pred_u],
    })

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Providers",    len(results))
    c2.metric("Predicted Fraud",    int((results['PredictedClass']=='Yes').sum()))
    c3.metric("Avg Fraud Prob",     f"{results['Probability'].mean():.3f}")

    st.dataframe(results.sort_values('Probability',ascending=False), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(results[results['PredictedClass']=='No']['Probability'],  bins=40, alpha=0.7, color=COLORS['legit'],  label='Legit')
    ax.hist(results[results['PredictedClass']=='Yes']['Probability'], bins=40, alpha=0.7, color=COLORS['fraud'], label='Fraud')
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel("Fraud Probability"); ax.legend()
    st.pyplot(fig); plt.close()

    if 'Label' in df_u.columns:
        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(df_u['Label'],y_pred_u):.3f}")
        c2.metric("ROC-AUC",  f"{roc_auc_score(df_u['Label'],y_proba_u):.3f}")
        c3.metric("F1-Score", f"{f1_score(df_u['Label'],y_pred_u):.3f}")

    st.download_button("⬇️ Download Predictions CSV",
        data=results.to_csv(index=False).encode(),
        file_name="Submission_Predictions.csv", mime="text/csv")

# ════════════════════════════════════════════════════════════════════
#  PAGE 6 — Recommendations
# ════════════════════════════════════════════════════════════════════
elif page == "💡 Business Recommendations":
    st.title("💡 Business Recommendations")

    st.subheader("🔍 Key Fraud Indicators")
    for title, desc in {
        "High Reimbursement Per Claim": "Fraudulent providers show significantly higher average reimbursement per claim.",
        "Unusual Admission Durations":  "Extreme short/long hospital stays indicate gaming the billing system.",
        "High Unique Physician Count":  "Billing under many physicians in short spans is a major red flag.",
        "Dead Patient Claims":          "High deceased beneficiary count suggests billing after patient death.",
        "Diagnosis Code Padding":       "Filling all 10 diagnosis code slots to justify expensive procedures.",
        "IP/OP Claim Ratio":            "Abnormally high inpatient vs outpatient ratio indicates upcoding.",
    }.items():
        with st.expander(f"⚠️ {title}"): st.write(desc)

    st.markdown("---")
    st.subheader("📋 Recommendations")
    for priority, rec in [
        ("🔴 High", "Flag providers in top 5% by ReimbursedPerClaim for immediate audit."),
        ("🔴 High", "Monitor providers with >200 unique patients but <5 physicians."),
        ("🟡 Medium", "Cross-check physician NPIs — duplicate or inactive physicians are suspicious."),
        ("🟡 Medium", "Alert when deceased beneficiary ratio > 10% for a provider."),
        ("🟢 Long Term", "Build temporal model tracking month-over-month claim changes."),
        ("🟢 Long Term", "Graph-based detection to uncover provider-physician fraud rings."),
    ]:
        c1, c2 = st.columns([1,5]); c1.markdown(priority); c2.info(rec)

    st.markdown("---")
    st.subheader("🚀 Model Improvement Roadmap")
    st.dataframe(pd.DataFrame({
        "Improvement":     ["SMOTE Balancing","XGBoost/LightGBM","Temporal Features","Graph Neural Networks","SHAP Explainability"],
        "Impact":          ["Medium","High","High","Very High","Medium"],
        "Effort":          ["Low","Low","Medium","High","Medium"],
        "Description":     [
            "Better class imbalance handling",
            "Often outperform RF on tabular fraud data",
            "Rolling 3/6-month provider claim windows",
            "Detect collusion fraud rings",
            "Explain individual provider predictions",
        ]
    }), use_container_width=True)
