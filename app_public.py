# --- Configuration and Limits ---
MAX_MOLECULES_LIMIT = 1000  # Configurable limit to prevent mining/abuse
RECOMMENDED_BATCH_SIZE = 100  # Recommended batch size for optimal performance
ENABLE_AD_CHECK = True  # Set True to compute nearest-training Tanimoto AD (slower)

import sys
import os

import warnings
import logging
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import json
import pickle
import time
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# Suppress Streamlit ScriptRunContext warnings from background threads
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.logger").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Streamlit page config must be the first Streamlit command in some runtimes.
st.set_page_config(
    page_title="KEAP1 IC50 Predictor | TCU",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
    RDKIT_CORE_AVAILABLE = True
    RDKIT_CORE_IMPORT_ERROR = ""
except ImportError as e:
    Chem = None
    DataStructs = None
    AllChem = None
    MACCSkeys = None
    rdFingerprintGenerator = None
    RDKIT_CORE_AVAILABLE = False
    RDKIT_CORE_IMPORT_ERROR = str(e)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
CWD_DIR = os.getcwd()


def _unique_paths(paths):
    unique = []
    seen = set()
    for p in paths:
        norm = os.path.abspath(p)
        if norm not in seen:
            unique.append(norm)
            seen.add(norm)
    return unique


SEARCH_ROOTS = _unique_paths([APP_DIR, REPO_DIR, CWD_DIR])

def app_path(*parts):
    return os.path.join(APP_DIR, *parts)


def find_existing_path(candidates, expect_dir=False):
    """Return the first existing path from a list of absolute candidates."""
    for candidate in candidates:
        if expect_dir and os.path.isdir(candidate):
            return candidate
        if (not expect_dir) and os.path.isfile(candidate):
            return candidate
    return None


def find_existing_file(rel_parts):
    """Find a file by relative path under app dir, repo root, then cwd."""
    candidates = [os.path.join(root, *rel_parts) for root in SEARCH_ROOTS]
    return find_existing_path(candidates, expect_dir=False)


def find_existing_dir(rel_parts):
    """Find a directory by relative path under app dir, repo root, then cwd."""
    candidates = [os.path.join(root, *rel_parts) for root in SEARCH_ROOTS]
    return find_existing_path(candidates, expect_dir=True)


def ensure_writable_dir(rel_parts):
    """Create a writable directory using search-root precedence."""
    for root in SEARCH_ROOTS:
        candidate = os.path.join(root, *rel_parts)
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except Exception:
            continue
    return os.path.join(APP_DIR, *rel_parts)


def resolve_descriptors_xml_path():
    """Resolve descriptors.xml for PaDEL in multiple deployment layouts."""
    candidates = []
    for root in SEARCH_ROOTS:
        candidates.extend([
            os.path.join(root, "descriptors.xml"),
            os.path.join(root, "app", "descriptors.xml"),
            os.path.join(root, "config", "descriptors.xml"),
            os.path.join(root, "app", "config", "descriptors.xml"),
        ])
    return find_existing_path(_unique_paths(candidates), expect_dir=False)

LOGS_DIR = ensure_writable_dir(("logs",))


@st.cache_resource
def load_training_descriptors_cached():
    """Load all training descriptors ONCE for AD checks (cached for reuse).
    
    Returns: (padel_df, rdkit_df, mordred_df) or (None, None, None) if not found
    """
    training_padel_df = None
    training_rdkit_df = None
    training_mordred_df = None
    
    # Try multiple possible paths across deployment layouts.
    possible_dirs = []
    for root in SEARCH_ROOTS:
        possible_dirs.extend([
            os.path.join(root, "features"),
            os.path.join(root, "results", "features"),
            os.path.join(root, "app", "features"),
            os.path.join(root, "app", "results", "features"),
        ])
    possible_dirs = _unique_paths(possible_dirs)
    
    feature_dir = None
    for pdir in possible_dirs:
        print(f"🔍 Checking for training features at: {pdir}")
        if os.path.exists(pdir):
            feature_dir = pdir
            print(f"✅ Found training features directory: {pdir}")
            break
    
    if not feature_dir:
        print("❌ Training features directory NOT found in any expected location")
        print(f"   APP_DIR = {APP_DIR}")
        print(f"   REPO_DIR = {REPO_DIR}")
        print(f"   CWD = {CWD_DIR}")
        return None, None, None
    
    try:
        print(f"📂 Loading training descriptors from: {feature_dir}")
        for fname in os.listdir(feature_dir):
            fpath = os.path.join(feature_dir, fname)
            if not fname.endswith('.csv'):
                continue
            
            fname_lower = fname.lower()
            try:
                if 'padel' in fname_lower and 'padel_descriptors.csv' in fname_lower:
                    print(f"   📥 Loading {fname}...")
                    training_padel_df = pd.read_csv(fpath, low_memory=False)
                    # ADD PREFIX TO MATCH SELECTED FEATURES
                    training_padel_df.columns = [f"PaDEL_{col}" for col in training_padel_df.columns]
                    print(f"      ✅ PaDEL loaded: {training_padel_df.shape[0]} rows × {training_padel_df.shape[1]} cols")
                elif 'rdkit' in fname_lower and 'rdkit_fingerprints.csv' in fname_lower:
                    print(f"   📥 Loading {fname}...")
                    training_rdkit_df = pd.read_csv(fpath, low_memory=False)
                    # ADD PREFIX TO MATCH SELECTED FEATURES
                    training_rdkit_df.columns = [f"RDKit_{col}" for col in training_rdkit_df.columns]
                    print(f"      ✅ RDKit loaded: {training_rdkit_df.shape[0]} rows × {training_rdkit_df.shape[1]} cols")
                elif 'mordred' in fname_lower and 'mordred_descriptors.csv' in fname_lower:
                    print(f"   📥 Loading {fname}...")
                    training_mordred_df = pd.read_csv(fpath, low_memory=False)
                    # ADD PREFIX TO MATCH SELECTED FEATURES
                    training_mordred_df.columns = [f"Mordred_{col}" for col in training_mordred_df.columns]
                    print(f"      ✅ Mordred loaded: {training_mordred_df.shape[0]} rows × {training_mordred_df.shape[1]} cols")
            except Exception as e:
                print(f"   ⚠️ Failed to load {fname}: {str(e)[:60]}")
    except Exception as e:
        print(f"❌ Error loading training descriptors: {str(e)[:100]}")
    
    print(f"📊 Training data summary: PaDEL={training_padel_df is not None}, RDKit={training_rdkit_df is not None}, Mordred={training_mordred_df is not None}")
    return training_padel_df, training_rdkit_df, training_mordred_df


def calculate_ad_tanimoto(query_padel_row, query_mordred_row, query_rdkit_row, selected_descriptor_names, training_padel_df, training_mordred_df, training_rdkit_df):
    """
    Calculate Tanimoto similarity using PRE-CALCULATED descriptors (from batch processing).
    
    Args:
        query_*_row: Pandas Series or DataFrame row with calculated descriptors for query molecule
        selected_descriptor_names: List of feature names to use for similarity calc
        training_*_df: Pre-loaded training descriptor DataFrames (cached)
    
    Returns:
        String: "0.892 (In-domain)" etc., or "N/A (reason)" if failed
    """
    if not selected_descriptor_names:
        return "N/A (no features)"
    
    # Check if training data exists
    if training_padel_df is None and training_mordred_df is None and training_rdkit_df is None:
        print("⚠️ AD Check: No training data loaded from cache, forcing refresh...")
        try:
            load_training_descriptors_cached.clear()
        except Exception:
            pass
        training_padel_df, training_mordred_df, training_rdkit_df = load_training_descriptors_cached()
        if training_padel_df is None and training_mordred_df is None and training_rdkit_df is None:
            print("⚠️ AD Check: No training data loaded after refresh!")
            return "N/A (no training data)"
    
    try:
        # Combine query descriptor rows into single row
        query_dfs = []
        if query_padel_row is not None:
            query_dfs.append(pd.DataFrame([query_padel_row]))
        if query_mordred_row is not None:
            query_dfs.append(pd.DataFrame([query_mordred_row]))
        if query_rdkit_row is not None:
            query_dfs.append(pd.DataFrame([query_rdkit_row]))
        
        if not query_dfs:
            print("⚠️ AD Check: No query descriptors!")
            return "N/A (no query desc)"
        
        query_features = pd.concat(query_dfs, axis=1)
        query_features = query_features.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        # Ensure all selected features are present (add missing at once)
        missing_features = [f for f in selected_descriptor_names if f not in query_features.columns]
        if missing_features:
            missing_df = pd.DataFrame(0.0, index=query_features.index, columns=missing_features)
            query_features = pd.concat([query_features, missing_df], axis=1)
        
        query_vector = query_features[selected_descriptor_names].values[0]
        
        # Check if query has ANY non-zero values
        if np.all(query_vector == 0):
            print("⚠️ AD Check: Query vector all zeros!")
            return "N/A (all zeros)"
        
        # Combine training descriptor DataFrames
        training_dfs = []
        if training_padel_df is not None:
            training_dfs.append(training_padel_df)
        if training_rdkit_df is not None:
            training_dfs.append(training_rdkit_df)
        if training_mordred_df is not None:
            training_dfs.append(training_mordred_df)
        
        if not training_dfs:
            print("⚠️ AD Check: Training dataframes empty!")
            return "N/A (no train desc)"
        
        training_features = pd.concat(training_dfs, axis=1)
        training_features = training_features.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        # Ensure selected features exist in training data (add missing at once)
        missing_train_features = [f for f in selected_descriptor_names if f not in training_features.columns]
        if missing_train_features:
            missing_train_df = pd.DataFrame(0.0, index=training_features.index, columns=missing_train_features)
            training_features = pd.concat([training_features, missing_train_df], axis=1)
        
        training_vectors = training_features[selected_descriptor_names].values
        
        if len(training_vectors) == 0:
            print("⚠️ AD Check: No training vectors!")
            return "N/A (no train vectors)"
        
        # Calculate generalized Tanimoto similarity in descriptor space.
        # For non-negative vectors a and b:
        # Tanimoto(a,b) = (a·b) / (||a||^2 + ||b||^2 - a·b)
        query_vector = np.asarray(query_vector, dtype=float)
        training_vectors = np.asarray(training_vectors, dtype=float)

        # Guard against negative numeric noise; descriptors are expected non-negative for this AD metric.
        query_vector = np.clip(query_vector, 0.0, None)
        training_vectors = np.clip(training_vectors, 0.0, None)

        dot_products = np.dot(training_vectors, query_vector)
        query_sq_norm = float(np.dot(query_vector, query_vector))
        train_sq_norms = np.einsum('ij,ij->i', training_vectors, training_vectors)
        denominators = train_sq_norms + query_sq_norm - dot_products

        # Avoid division by zero; similarity is 0 where denominator is zero.
        sims = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products, dtype=float),
            where=denominators > 0
        )
        
        if len(sims) == 0:
            return "N/A (no sims)"
        
        max_sim = float(np.max(sims))
        print(f"✅ AD Check Success: max Tanimoto = {max_sim:.3f}")
        
        # Thresholds calibrated for Tanimoto similarity
        if max_sim >= 0.60:
            label = "In-domain"
        elif max_sim >= 0.40:
            label = "Borderline"
        else:
            label = "Out-of-domain"
        
        return f"{max_sim:.3f} ({label})"
    
    except Exception as e:
        print(f"❌ AD Check Error: {str(e)[:60]}")
        return f"N/A (err: {str(e)[:15]})"

# NumPy 2.x compatibility shim for Mordred, which still imports numpy.product.
if not hasattr(np, 'product'):
    np.product = np.prod

try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
    print("✅ Mordred descriptors available")
except ImportError as e:
    print(f"⚠️ Mordred import failed: {e}")
    print("ℹ️ Mordred descriptors will be disabled due to NumPy compatibility")
    MORDRED_AVAILABLE = False
    Calculator = None
    descriptors = None

# Import boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


try:
    from padelpy import padeldescriptor
    PADEL_AVAILABLE = True  
    print("✅ PaDEL descriptors available")
except ImportError:
    PADEL_AVAILABLE = False
    print("❌ PaDEL descriptors not available")

# Avalon fingerprints (part of RDKit contrib)
try:
    from rdkit.Avalon import pyAvalonTools
    AVALON_AVAILABLE = True
    print("✅ Avalon fingerprints available")
except ImportError:
    AVALON_AVAILABLE = False
    print("⚠️ Avalon fingerprints not available (rdkit.Avalon missing)")

# Mol2Vec embeddings
try:
    from mol2vec.features import mol2alt_sentence, sentences2vec
    from gensim.models import Word2Vec as GensimWord2Vec
    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False

# MAP4 fingerprints
try:
    from map4 import MAP4Calculator
    MAP4_AVAILABLE = True
except ImportError:
    MAP4_AVAILABLE = False

# RDKit Scaffold (always available if RDKit is)
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    SCAFFOLD_AVAILABLE = True
    print("✅ Scaffold fingerprints available")
except ImportError:
    SCAFFOLD_AVAILABLE = False
    print("⚠️ MurckoScaffold not available")

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class GlobalTimer:
    def __init__(self):
        self.start_time = None
        self.is_running = False
    
    def start(self):
        """Start the independent timer"""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
    
    def stop(self):
        """Stop the timer"""
        self.is_running = False
    
    def get_elapsed_time(self):
        """Get current elapsed time in seconds"""
        if self.start_time and self.is_running:
            return time.time() - self.start_time
        return 0
    
    def format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Global timer instance
if 'global_timer' not in st.session_state:
    st.session_state.global_timer = GlobalTimer()

# --- Load TCU logo ---
logo_path = find_existing_file(("tcu_logo.png",)) or find_existing_file(("app", "tcu_logo.png"))
logo = None
if logo_path:
    try:
        logo = Image.open(logo_path)
    except Exception as e:
        print(f"⚠️ Could not load logo at {logo_path}: {e}")

if not RDKIT_CORE_AVAILABLE:
    st.error("❌ RDKit is required but not installed in this deployment environment.")
    st.info("Install a compatible RDKit build in requirements and redeploy.")
    st.caption(f"Import error: {RDKIT_CORE_IMPORT_ERROR}")
    st.stop()

# --- TCU Custom CSS Theme ---
st.markdown("""
<style>
    /* TCU Purple Theme */
    :root {
        --tcu-purple: #4d1979;
        --tcu-light-purple: #663399;
        --tcu-silver: #c0c0c0;
        --tcu-dark-gray: #333333;
        --tcu-light-gray: #f5f5f5;
    }
    
    /* Main header styling */
    .big-font { 
        font-size: 32px !important; 
        font-weight: bold; 
        color: var(--tcu-purple);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 5px;
    }
    
    .subtext { 
        font-size: 18px; 
        color: var(--tcu-dark-gray);
        font-style: italic;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--tcu-light-gray);
    }
    
    /* Success messages */
    .stSuccess {
        background-color: rgba(77, 25, 121, 0.1);
        border: 1px solid var(--tcu-purple);
        color: var(--tcu-purple);
    }
    
    /* Info messages */
    .stInfo {
        background-color: rgba(77, 25, 121, 0.05);
        border-left: 4px solid var(--tcu-purple);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--tcu-purple);
    }
    
    /* Metrics */
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid var(--tcu-silver);
        box-shadow: 0 2px 4px rgba(77, 25, 121, 0.1);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed var(--tcu-silver);
        border-radius: 10px;
        padding: 20px;
        background-color: rgba(77, 25, 121, 0.02);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--tcu-purple);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--tcu-light-purple);
        box-shadow: 0 4px 8px rgba(77, 25, 121, 0.3);
    }
    
    /* Download button styling */
    .download-btn {
        background-color: var(--tcu-purple) !important;
        color: white !important;
        padding: 12px 24px !important;
        text-decoration: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        display: inline-block !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(77, 25, 121, 0.2) !important;
    }
    
    .download-btn:hover {
        background-color: var(--tcu-light-purple) !important;
        box-shadow: 0 4px 8px rgba(77, 25, 121, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Table styling */
    .stDataFrame {
        border: 2px solid var(--tcu-silver);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Error highlighting for table */
    .error-cell {
        background-color: #ffebee !important;
        color: #c62828 !important;
    }
    
    /* Section headers */
    h3 {
        color: var(--tcu-purple) !important;
        border-bottom: 2px solid var(--tcu-silver);
        padding-bottom: 5px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: var(--tcu-dark-gray);
        font-size: 14px;
        background-color: var(--tcu-light-gray);
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        border: 1px solid var(--tcu-silver);
    }
    
    .footer strong {
        color: var(--tcu-purple);
    }
</style>
""", unsafe_allow_html=True)


# --- Title and Branding ---
col1, col2 = st.columns([1, 6])

with col1:
    if logo is not None:
        st.image(logo, width=120)
    else:
        st.markdown("### TCU")

with col2:
    st.markdown("""
    <div style="margin-left: -20px;">
        <div class="big-font">🏛️ Texas Christian University | KEAP1 IC50 Predictor</div>
        <div class="subtext">Department of Chemistry and Biochemistry </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Mordred Calculator ---
if MORDRED_AVAILABLE:
    mordred_calc = Calculator(descriptors, ignore_3D=True)
else:
    mordred_calc = None

# --- Helper Functions ---
def get_mordred_features(smiles):
    """Calculate Mordred descriptors from SMILES.
    Returns: tuple (dataframe, descriptor_quality_dict)
    """
    descriptor_quality = {
        'mordred': {'calculated': False, 'failed': False}
    }
    
    if not MORDRED_AVAILABLE:
        descriptor_quality['mordred']['failed'] = True
        print("⚠️ Mordred descriptors not available - NumPy compatibility issues")
        return (None, descriptor_quality)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (None, descriptor_quality)
    try:
        from rdkit.Chem import rdMolDescriptors, Descriptors as _Desc
        desc = mordred_calc(mol).asdict()
        numeric_desc = {}
        for key, value in desc.items():
            try:
                numeric_value = float(value)
                if np.isinf(numeric_value) or np.isnan(numeric_value):
                    numeric_desc[key] = np.nan
                else:
                    numeric_desc[key] = numeric_value
            except (ValueError, TypeError, OverflowError):
                numeric_desc[key] = np.nan

        # Patch descriptors removed/renamed in newer Mordred versions so
        # the feature set matches the training scaler exactly.

        # FCSP3: fraction of sp3 carbons (exact RDKit equivalent)
        if 'FCSP3' not in numeric_desc:
            try:
                numeric_desc['FCSP3'] = float(rdMolDescriptors.CalcFractionCSP3(mol))
            except Exception:
                numeric_desc['FCSP3'] = np.nan

        # nHetero: number of heteroatoms (exact RDKit equivalent)
        if 'nHetero' not in numeric_desc:
            try:
                numeric_desc['nHetero'] = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
            except Exception:
                numeric_desc['nHetero'] = np.nan

        # FilterItLogS: aqueous solubility estimate (Delaney/ESOL formula
        # used by the old Filter-It algorithm embedded in earlier Mordred)
        if 'FilterItLogS' not in numeric_desc:
            try:
                _logp = _Desc.MolLogP(mol)
                _mw   = _Desc.MolWt(mol)
                _rb   = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
                _nAt  = mol.GetNumAtoms()
                _nAr  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
                _ap   = _nAr / _nAt if _nAt > 0 else 0.0
                numeric_desc['FilterItLogS'] = (
                    0.16 - 0.63 * _logp - 0.0062 * _mw + 0.066 * _rb - 0.74 * _ap
                )
            except Exception:
                numeric_desc['FilterItLogS'] = np.nan

        df = pd.DataFrame([numeric_desc])
        df.columns = [f'Mordred_{str(col)}' for col in df.columns]
        df.columns = df.columns.astype(str)
        descriptor_quality['mordred']['calculated'] = True
        return (df, descriptor_quality)
    except Exception as e:
        descriptor_quality['mordred']['failed'] = True
        print(f"⚠️ Mordred calculation failed for SMILES {smiles[:50]}...: {str(e)[:30]}")
        return (None, descriptor_quality)

def _make_morgan_generator(radius, fp_size, use_counts=False):
    """Mirror of make_morgan_generator from 04_fingerprinting.py — uses same API priority."""
    from rdkit.Chem import rdFingerprintGenerator
    if hasattr(rdFingerprintGenerator, 'MorganGenerator'):
        try:
            return rdFingerprintGenerator.MorganGenerator(
                radius=radius, fpSize=fp_size, useCounts=use_counts)
        except TypeError:
            return rdFingerprintGenerator.MorganGenerator(
                radius=radius, fpSize=fp_size, countSimulation=use_counts)
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size, countSimulation=use_counts)


def get_rdkit_features(smiles, required_features=None):
    """Calculate RDKit features using the same schema as 04_fingerprinting.py.
    Returns: tuple (dataframe, descriptor_quality_dict)
    """
    descriptor_quality = {
        'rdkit_morgan': {'calculated': False, 'failed': False},
        'rdkit_morgan6': {'calculated': False, 'failed': False},
        'rdkit_morgancount': {'calculated': False, 'failed': False},
        'rdkit_torsion': {'calculated': False, 'failed': False},
        'rdkit_avalon': {'calculated': False, 'failed': False},
        'rdkit_fp': {'calculated': False, 'failed': False},
        'rdkit_scaffold': {'calculated': False, 'failed': False},
        'rdkit_descriptors': {'calculated': False, 'failed': False},
        'all_calculated': True
    }
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        all_arrays = []
        all_names = []

        required_set = set(required_features) if required_features else None

        def _family_needed(prefix):
            if required_set is None:
                return True
            return any(f.startswith(prefix) for f in required_set)

        # 1. Morgan ECFP4 (2048 bits) — radius=2, binary
        morgan_size = 2048
        if _family_needed('RDKit_Morgan_'):
            gen_morgan = _make_morgan_generator(radius=2, fp_size=morgan_size, use_counts=False)
            morgan_fp = gen_morgan.GetFingerprintAsNumPy(mol).astype(float)
            descriptor_quality['rdkit_morgan']['calculated'] = True
            all_arrays.append(morgan_fp)
            all_names += [f'RDKit_Morgan_{i}' for i in range(morgan_size)]

        # 2. Morgan ECFP6 (2048 bits) — radius=3, binary
        if _family_needed('RDKit_Morgan6_'):
            gen_morgan6 = _make_morgan_generator(radius=3, fp_size=morgan_size, use_counts=False)
            morgan6_fp = gen_morgan6.GetFingerprintAsNumPy(mol).astype(float)
            descriptor_quality['rdkit_morgan6']['calculated'] = True
            all_arrays.append(morgan6_fp)
            all_names += [f'RDKit_Morgan6_{i}' for i in range(morgan_size)]

        # 3. Morgan Count (2048 features)
        if _family_needed('RDKit_MorganCount_'):
            gen_mcount = _make_morgan_generator(radius=2, fp_size=morgan_size, use_counts=True)
            sparse_count = gen_mcount.GetCountFingerprint(mol)
            mcount_arr = np.zeros(morgan_size)
            for k, v in sparse_count.GetNonzeroElements().items():
                mcount_arr[k % morgan_size] += v
            mcount_fp = mcount_arr.astype(float)
            descriptor_quality['rdkit_morgancount']['calculated'] = True
            all_arrays.append(mcount_fp)
            all_names += [f'RDKit_MorganCount_{i}' for i in range(morgan_size)]

        # 4. Topological Torsion (1024 bits)
        torsion_size = 1024
        if _family_needed('RDKit_Torsion_'):
            gen_tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=torsion_size)
            tt_fp = gen_tt.GetFingerprintAsNumPy(mol).astype(float)
            descriptor_quality['rdkit_torsion']['calculated'] = True
            all_arrays.append(tt_fp)
            all_names += [f'RDKit_Torsion_{i}' for i in range(torsion_size)]

        # 5. Avalon fingerprints (1024 bits)
        avalon_size = 1024
        if _family_needed('RDKit_Avalon_'):
            if not AVALON_AVAILABLE:
                raise ImportError("Avalon fingerprints not available. Install with: pip install rdkit-pypi")
            avalon_fp = np.array(pyAvalonTools.GetAvalonFP(mol, nBits=avalon_size), dtype=float)
            descriptor_quality['rdkit_avalon']['calculated'] = True
            all_arrays.append(avalon_fp)
            all_names += [f'RDKit_Avalon_{i}' for i in range(avalon_size)]

        # 6. RDKit native fingerprint (1024 bits) — generator API, same as training
        rdkitfp_size = 1024
        if _family_needed('RDKit_RDKitFP_'):
            gen_rdkfp = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=rdkitfp_size)
            rdkfp = gen_rdkfp.GetFingerprintAsNumPy(mol).astype(float)
            descriptor_quality['rdkit_fp']['calculated'] = True
            all_arrays.append(rdkfp)
            all_names += [f'RDKit_RDKitFP_{i}' for i in range(rdkitfp_size)]

        # 7. Scaffold Morgan fingerprint (512 bits)
        scaffold_size = 512
        if _family_needed('RDKit_Scaffold_'):
            scaffold_core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_gen = _make_morgan_generator(radius=2, fp_size=scaffold_size, use_counts=False)
            scaffold_fp = scaffold_gen.GetFingerprintAsNumPy(scaffold_core).astype(float)
            descriptor_quality['rdkit_scaffold']['calculated'] = True
            all_arrays.append(scaffold_fp)
            all_names += [f'RDKit_Scaffold_{i}' for i in range(scaffold_size)]

        # 8. RDKit molecular descriptors
        desc_vals = []
        desc_names = []
        needed_desc = None
        if required_set is not None:
            needed_desc = {
                f[len('RDKit_RDKit_'):]
                for f in required_set
                if f.startswith('RDKit_RDKit_')
            }
        for desc_name, fn in Descriptors.descList:
            if needed_desc is not None and desc_name not in needed_desc:
                continue
            try:
                value = fn(mol)
                if value is None or np.isinf(value) or np.isnan(value):
                    desc_vals.append(np.nan)
                else:
                    desc_vals.append(float(value))
            except Exception:
                desc_vals.append(np.nan)
            desc_names.append(f'RDKit_RDKit_{desc_name}')

        if desc_names:
            desc_array = np.array(desc_vals, dtype=float)
            all_arrays.append(desc_array)
            all_names += desc_names

        if not all_arrays:
            return (pd.DataFrame(index=[0]), descriptor_quality)

        # Combine all into DataFrame
        combined = np.concatenate(all_arrays).astype(float)
        rdkit_df = pd.DataFrame([combined], columns=all_names)
        rdkit_df.columns = rdkit_df.columns.astype(str)
        rdkit_df = rdkit_df.fillna(0.0)
        rdkit_df.replace([np.inf, -np.inf], 0.0, inplace=True)

        return (rdkit_df, descriptor_quality)

    except ImportError:
        print("❌ RDKit not available")
        return (None, {k: {'calculated': False, 'failed': True} for k in ['rdkit_morgan', 'rdkit_morgan6', 'rdkit_morgancount', 'rdkit_torsion', 'rdkit_avalon', 'rdkit_fp', 'rdkit_scaffold', 'rdkit_descriptors']})
    except Exception as e:
        print(f"❌ RDKit calculation error: {e}")
        return (None, {k: {'calculated': False, 'failed': True} for k in ['rdkit_morgan', 'rdkit_morgan6', 'rdkit_morgancount', 'rdkit_torsion', 'rdkit_avalon', 'rdkit_fp', 'rdkit_scaffold', 'rdkit_descriptors']})

def get_padel_features(smiles):
    """Extract PaDEL descriptors from SMILES - enhanced with retry logic for Windows file locking issues.
    Returns: tuple (dataframe, descriptor_quality_dict)
    """
    descriptor_quality = {
        'padel': {'calculated': False, 'failed': False}
    }
    
    if not PADEL_AVAILABLE:
        descriptor_quality['padel']['failed'] = True
        return (None, descriptor_quality)
    
    try:
        import os
        import time
        import random
        from rdkit import Chem
        
        # First, validate SMILES with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Silent failure for invalid SMILES
        
        # Enhanced file naming with process ID and random component to avoid conflicts
        timestamp = str(int(time.time() * 1000))
        random_id = str(random.randint(1000, 9999))
        process_id = str(os.getpid())
        temp_dir = tempfile.gettempdir()
        smi_file = os.path.join(temp_dir, f"padel_temp_{process_id}_{timestamp}_{random_id}.smi")
        csv_file = os.path.join(temp_dir, f"padel_temp_{process_id}_{timestamp}_{random_id}.csv")
        
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Ensure files are fully closed before PaDEL access
                with open(smi_file, 'w', encoding='utf-8') as f:
                    f.write(f"{smiles}\tmol1\n")
                    f.flush()  # Force write to disk
                    try:
                        os.fsync(f.fileno())  # Ensure OS writes to disk
                    except OSError:
                        pass
                
                # Brief pause for file system stability (prevents file locking)
                time.sleep(0.05)
                
                # Calculate PaDEL descriptors using the same descriptor XML strategy as 04_fingerprinting.py
                descriptor_xml_path = resolve_descriptors_xml_path()

                padel_kwargs = dict(
                    mol_dir=smi_file,
                    d_file=csv_file,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=1,
                    removesalt=True,
                    log=False,
                    fingerprints=True,
                    maxruntime=30000
                )
                if descriptor_xml_path is not None:
                    padel_kwargs['descriptortypes'] = descriptor_xml_path

                padeldescriptor(**padel_kwargs)
                
                # Wait for output file and verify it's complete
                max_wait = 5  # seconds
                wait_time = 0
                while wait_time < max_wait:
                    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                        time.sleep(0.1)  # Brief wait to ensure file write is complete
                        break
                    time.sleep(0.2)  # Check every 0.2 seconds
                    wait_time += 0.2
                
                # Read and process results
                if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                    try:
                        padel_df = pd.read_csv(csv_file)
                        
                        if len(padel_df) > 0:
                            # Remove Name column if present
                            if 'Name' in padel_df.columns:
                                padel_df = padel_df.drop('Name', axis=1)
                            
                            # Ensure column names are strings
                            padel_df.columns = [f'PaDEL_{str(col)}' for col in padel_df.columns]
                            padel_df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in padel_df.columns]
                            padel_df.columns = padel_df.columns.astype(str)
                            
                            # Convert to numeric and handle errors
                            padel_df = padel_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                            
                            descriptor_quality['padel']['calculated'] = True
                            return (padel_df, descriptor_quality)  # Success!
                        
                    except Exception as read_error:
                        if attempt < max_retries - 1:
                            continue  # Retry on read error
                        else:
                            descriptor_quality['padel']['failed'] = True
                            print(f"⚠️ PaDEL read error for SMILES {smiles[:50]}...: {str(read_error)[:30]}")
                            return (None, descriptor_quality)  # Final attempt failed
                
                # If we get here, the output wasn't created properly
                if attempt < max_retries - 1:
                    time.sleep(0.2)  # Small delay before retry to prevent file conflicts
                    continue
                else:
                    return None  # All attempts failed
                    
            except Exception as e:
                if "FileNotFoundException" in str(e) or "being used by another process" in str(e):
                    if attempt < max_retries - 1:
                        # File locking issue - wait briefly before retry
                        time.sleep(0.3)  # Essential delay to resolve file locking
                        continue
                    else:
                        return None  # Final retry failed
                else:
                    # Other error - don't retry
                    return None
                    
            finally:
                # Enhanced cleanup with retry logic
                cleanup_files = [smi_file, csv_file]
                for file_path in cleanup_files:
                    for cleanup_attempt in range(3):
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                break  # Success
                        except (PermissionError, FileNotFoundError):
                            if cleanup_attempt < 2:
                                time.sleep(0.1)  # Small delay for file cleanup
                            # Ignore cleanup failures on final attempt
        
        descriptor_quality['padel']['failed'] = True
        print(f"⚠️ PaDEL all retries exhausted for SMILES {smiles[:50]}...")
        return (None, descriptor_quality)  # All retries exhausted
        
    except Exception as e:
        descriptor_quality['padel']['failed'] = True
        print(f"⚠️ PaDEL error for SMILES {smiles[:50]}...: {str(e)[:30]}")
        return (None, descriptor_quality)  # Silent failure for any other issues

def get_padel_features_batch(smiles_list):
    """Extract PaDEL descriptors from multiple SMILES at once - optimized batch processing"""
    if not PADEL_AVAILABLE or not smiles_list:
        return None
    
    try:
        import os
        import time
        import random
        from rdkit import Chem
        
        # First, validate all SMILES with RDKit and filter out invalid ones
        valid_smiles = []
        valid_indices = []
        invalid_count = 0
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumAtoms() > 0:
                    valid_smiles.append(smiles)
                    valid_indices.append(i)
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
        
        if not valid_smiles:
            print(f"Warning: All {len(smiles_list)} SMILES are invalid for PaDEL processing")
            return None
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid SMILES skipped in PaDEL batch processing")
        
        # Enhanced file naming with process ID and random component to avoid conflicts
        timestamp = str(int(time.time() * 1000))
        random_id = str(random.randint(1000, 9999))
        process_id = str(os.getpid())
        temp_dir = tempfile.gettempdir()
        smi_file = os.path.join(temp_dir, f"padel_batch_temp_{process_id}_{timestamp}_{random_id}.smi")
        csv_file = os.path.join(temp_dir, f"padel_batch_temp_{process_id}_{timestamp}_{random_id}.csv")
        
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Create batch SMI file with all valid SMILES
                with open(smi_file, 'w', encoding='utf-8') as f:
                    for i, smiles in enumerate(valid_smiles):
                        f.write(f"{smiles}\tmol{i+1}\n")
                    f.flush()  # Force write to disk
                    try:
                        os.fsync(f.fileno())  # Ensure OS writes to disk
                    except OSError:
                        pass
                
                # Brief pause for file system stability (prevents file locking)
                time.sleep(0.05)
                
                descriptor_xml_path = resolve_descriptors_xml_path()

                if descriptor_xml_path is None:
                    print("Warning: descriptors.xml not found, using default PaDEL descriptors")

                padel_kwargs = dict(
                    mol_dir=smi_file,
                    d_file=csv_file,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=min(4, len(valid_smiles)),  # Careful threading
                    removesalt=True,
                    log=False,
                    fingerprints=True,
                    maxruntime=60000,  # Increased timeout for batch processing
                    retain3d=False,  # Ensure 2D only
                    retainorder=True  # Maintain molecule order
                )
                if descriptor_xml_path is not None:
                    padel_kwargs['descriptortypes'] = descriptor_xml_path

                padeldescriptor(**padel_kwargs)
                
                # Wait for output file and verify it's complete
                max_wait = 15  # seconds for batch processing
                wait_time = 0
                while wait_time < max_wait:
                    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                        # Additional check: ensure file is not still being written
                        time.sleep(0.2)
                        if os.path.exists(csv_file) and os.path.getsize(csv_file) == os.path.getsize(csv_file):  # Size stable
                            break
                    time.sleep(0.5)
                    wait_time += 0.5
                
                # Read and process results
                if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                    try:
                        padel_df = pd.read_csv(csv_file, low_memory=False)
                        
                        if len(padel_df) > 0 and len(padel_df.columns) > 1:  # Ensure we have descriptors
                            # Remove Name column if present
                            if 'Name' in padel_df.columns:
                                padel_df = padel_df.drop('Name', axis=1)
                            
                            # Validate that we have meaningful descriptors
                            if len(padel_df.columns) < 10:  # Suspiciously few descriptors
                                print(f"Warning: Only {len(padel_df.columns)} PaDEL descriptors generated, expected hundreds")
                                if attempt < max_retries - 1:
                                    continue
                            
                            # Ensure column names are strings and properly prefixed
                            padel_df.columns = [f'PaDEL_{str(col)}' for col in padel_df.columns]
                            padel_df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '') for col in padel_df.columns]
                            padel_df.columns = padel_df.columns.astype(str)
                            
                            # Convert to numeric carefully
                            padel_df = padel_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                            
                            # Validate final dataframe
                            if padel_df.shape[0] != len(valid_smiles):
                                print(f"Warning: PaDEL output has {padel_df.shape[0]} rows, expected {len(valid_smiles)}")
                            
                            # Reorder to match original SMILES order, filling missing with NaN
                            result_df = pd.DataFrame(index=range(len(smiles_list)), columns=padel_df.columns)
                            for orig_idx, batch_idx in enumerate(valid_indices):
                                if batch_idx < len(padel_df):
                                    result_df.iloc[orig_idx] = padel_df.iloc[batch_idx]
                            
                            print(f"Successfully generated {len(result_df.columns)} PaDEL descriptors for {len(valid_smiles)} valid molecules")
                            return result_df  # Success!
                        
                    except Exception as read_error:
                        print(f"Error reading PaDEL output: {read_error}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return None  # Final attempt failed
                
                # If we get here, the output wasn't created properly
                if attempt < max_retries - 1:
                    time.sleep(0.2)  # Small delay before retry to prevent file conflicts
                    continue
                else:
                    return None  # All attempts failed
                    
            except Exception as e:
                if "FileNotFoundException" in str(e) or "being used by another process" in str(e):
                    if attempt < max_retries - 1:
                        # File locking issue - wait briefly before retry
                        time.sleep(0.3)  # Essential delay to resolve file locking
                        continue
                    else:
                        return None  # Final retry failed
                else:
                    # Other error - don't retry
                    return None
                    
            finally:
                # Enhanced cleanup with retry logic
                cleanup_files = [smi_file, csv_file]
                for file_path in cleanup_files:
                    for cleanup_attempt in range(3):
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                break  # Success
                        except (PermissionError, FileNotFoundError):
                            if cleanup_attempt < 2:
                                time.sleep(0.1)  # Small delay for file cleanup
                            # Ignore cleanup failures on final attempt
        
        return None  # All retries exhausted
        
    except Exception as e:
        return None  # Silent failure for any other issues

def get_mordred_features_batch(smiles_list):
    """Extract Mordred descriptors from multiple SMILES at once - optimized batch processing"""
    if not MORDRED_AVAILABLE:
        return None
    
    try:
        from rdkit.Chem import rdMolDescriptors, Descriptors as _Desc
        
        # Validate all SMILES and create molecules
        valid_mols = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_indices.append(i)
        
        if not valid_mols:
            return None
        
        # Calculate Mordred descriptors for all molecules
        desc_results = []
        for i, smiles in enumerate(smiles_list):
            if i in valid_indices:
                # This molecule is valid, get its position in valid_mols
                mol_idx = valid_indices.index(i)
                mol = valid_mols[mol_idx]
                try:
                    desc = mordred_calc(mol).asdict()
                    numeric_desc = {}
                    for key, value in desc.items():
                        try:
                            numeric_value = float(value)
                            if np.isinf(numeric_value) or np.isnan(numeric_value):
                                numeric_desc[key] = np.nan
                            else:
                                numeric_desc[key] = numeric_value
                        except (ValueError, TypeError, OverflowError):
                            numeric_desc[key] = np.nan
                    
                    # Apply patches for compatibility
                    # FCSP3: fraction of sp3 carbons
                    if 'FCSP3' not in numeric_desc:
                        try:
                            numeric_desc['FCSP3'] = float(rdMolDescriptors.CalcFractionCSP3(mol))
                        except Exception:
                            numeric_desc['FCSP3'] = np.nan
                    
                    # nHetero: number of heteroatoms
                    if 'nHetero' not in numeric_desc:
                        try:
                            numeric_desc['nHetero'] = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
                        except Exception:
                            numeric_desc['nHetero'] = np.nan
                    
                    # FilterItLogS: aqueous solubility estimate
                    if 'FilterItLogS' not in numeric_desc:
                        try:
                            _logp = _Desc.MolLogP(mol)
                            _mw   = _Desc.MolWt(mol)
                            _rb   = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
                            _nAt  = mol.GetNumAtoms()
                            _nAr  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
                            _ap   = _nAr / _nAt if _nAt > 0 else 0.0
                            numeric_desc['FilterItLogS'] = (
                                0.16 - 0.63 * _logp - 0.0062 * _mw + 0.066 * _rb - 0.74 * _ap
                            )
                        except Exception:
                            numeric_desc['FilterItLogS'] = np.nan
                    
                    desc_results.append(numeric_desc)
                except Exception:
                    # For failed molecules, create empty dict with NaN values
                    desc_results.append({})
            else:
                # Invalid molecule
                desc_results.append({})
        
        # Create DataFrame with all results (same length as input)
        mordred_df = pd.DataFrame(desc_results)
        mordred_df.columns = [f'Mordred_{str(col)}' for col in mordred_df.columns]
        mordred_df.columns = mordred_df.columns.astype(str)
        
        # Reorder to match original SMILES order, filling missing with NaN
        result_df = pd.DataFrame(index=range(len(smiles_list)), columns=mordred_df.columns)
        for orig_idx, batch_idx in enumerate(valid_indices):
            if batch_idx < len(mordred_df):
                result_df.iloc[orig_idx] = mordred_df.iloc[batch_idx]
        
        return result_df
        
    except Exception as e:
        return None

def get_rdkit_features_batch(smiles_list, required_features=None):
    """Calculate RDKit features for multiple SMILES at once - optimized batch processing with careful validation"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # Validate all SMILES and create molecules
        valid_mols = []
        valid_indices = []
        invalid_count = 0
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms() > 0:
                valid_mols.append(mol)
                valid_indices.append(i)
            else:
                invalid_count += 1
        
        if not valid_mols:
            print(f"No valid molecules found in batch of {len(smiles_list)} SMILES")
            return None
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid SMILES strings skipped")
        
        required_set = set(required_features) if required_features else None
        
        def _family_needed(prefix):
            if required_set is None:
                return True  # Generate all features if no requirements specified
            return any(f.startswith(prefix) for f in required_set)
        
        # Determine which features are needed
        needs_morgan = _family_needed('RDKit_Morgan_')
        needs_morgan6 = _family_needed('RDKit_Morgan6_')
        needs_morgancount = _family_needed('RDKit_MorganCount_')
        needs_torsion = _family_needed('RDKit_Torsion_')
        needs_avalon = _family_needed('RDKit_Avalon_')
        needs_rdkfp = _family_needed('RDKit_RDKitFP_')
        needs_scaffold = _family_needed('RDKit_Scaffold_')
        needs_descriptors = _family_needed('RDKit_RDKit_')
        
        # Prepare generators once
        morgan_size = 2048
        gen_morgan = _make_morgan_generator(radius=2, fp_size=morgan_size, use_counts=False) if needs_morgan else None
        gen_morgan6 = _make_morgan_generator(radius=3, fp_size=morgan_size, use_counts=False) if needs_morgan6 else None
        gen_mcount = _make_morgan_generator(radius=2, fp_size=morgan_size, use_counts=True) if needs_morgancount else None
        
        torsion_size = 1024
        gen_tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=torsion_size) if needs_torsion else None
        
        avalon_size = 1024
        rdkitfp_size = 1024
        gen_rdkfp = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=rdkitfp_size) if needs_rdkfp else None
        
        scaffold_size = 512
        
        # Determine needed descriptors
        needed_desc_names = None
        if required_set is not None and needs_descriptors:
            needed_desc_names = {
                f[len('RDKit_RDKit_'):].replace('_', '')
                for f in required_set
                if f.startswith('RDKit_RDKit_')
            }
        
        # Process all molecules
        all_results = []
        processing_errors = 0
        
        for mol in valid_mols:
            all_arrays = []
            all_names = []
            
            try:
                # 1. Morgan ECFP4
                if needs_morgan:
                    try:
                        morgan_fp = gen_morgan.GetFingerprintAsNumPy(mol).astype(float)
                    except Exception:
                        morgan_fp = np.zeros(morgan_size)
                    all_arrays.append(morgan_fp)
                    all_names += [f'RDKit_Morgan_{i}' for i in range(morgan_size)]
                
                # 2. Morgan ECFP6
                if needs_morgan6:
                    try:
                        morgan6_fp = gen_morgan6.GetFingerprintAsNumPy(mol).astype(float)
                    except Exception:
                        morgan6_fp = np.zeros(morgan_size)
                    all_arrays.append(morgan6_fp)
                    all_names += [f'RDKit_Morgan6_{i}' for i in range(morgan_size)]
                
                # 3. Morgan Count
                if needs_morgancount:
                    try:
                        sparse_count = gen_mcount.GetCountFingerprint(mol)
                        mcount_arr = np.zeros(morgan_size)
                        for k, v in sparse_count.GetNonzeroElements().items():
                            mcount_arr[k % morgan_size] += v
                        mcount_fp = mcount_arr.astype(float)
                    except Exception:
                        mcount_fp = np.zeros(morgan_size)
                    all_arrays.append(mcount_fp)
                    all_names += [f'RDKit_MorganCount_{i}' for i in range(morgan_size)]
                
                # 4. Topological Torsion
                if needs_torsion:
                    try:
                        tt_fp = gen_tt.GetFingerprintAsNumPy(mol).astype(float)
                    except Exception:
                        tt_fp = np.zeros(torsion_size)
                    all_arrays.append(tt_fp)
                    all_names += [f'RDKit_Torsion_{i}' for i in range(torsion_size)]
                
                # 5. Avalon fingerprints
                if needs_avalon:
                    if AVALON_AVAILABLE:
                        try:
                            avalon_fp = np.array(pyAvalonTools.GetAvalonFP(mol, nBits=avalon_size), dtype=float)
                        except Exception:
                            avalon_fp = np.zeros(avalon_size)
                    else:
                        avalon_fp = np.zeros(avalon_size)
                    all_arrays.append(avalon_fp)
                    all_names += [f'RDKit_Avalon_{i}' for i in range(avalon_size)]
                
                # 6. RDKit native fingerprint
                if needs_rdkfp:
                    try:
                        rdkfp = gen_rdkfp.GetFingerprintAsNumPy(mol).astype(float)
                    except Exception:
                        rdkfp = np.zeros(rdkitfp_size)
                    all_arrays.append(rdkfp)
                    all_names += [f'RDKit_RDKitFP_{i}' for i in range(rdkitfp_size)]
                
                # 7. Scaffold Morgan fingerprint
                if needs_scaffold:
                    try:
                        scaffold_core = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_gen = _make_morgan_generator(radius=2, fp_size=scaffold_size, use_counts=False)
                        scaffold_fp = scaffold_gen.GetFingerprintAsNumPy(scaffold_core).astype(float)
                    except Exception:
                        scaffold_fp = np.zeros(scaffold_size)
                    all_arrays.append(scaffold_fp)
                    all_names += [f'RDKit_Scaffold_{i}' for i in range(scaffold_size)]
                
                # 8. RDKit molecular descriptors
                if needs_descriptors:
                    desc_vals = []
                    desc_names = []
                    for desc_name, fn in Descriptors.descList:
                        if needed_desc_names is not None and desc_name not in needed_desc_names:
                            continue
                        try:
                            value = fn(mol)
                            if value is None or np.isinf(value) or np.isnan(value):
                                desc_vals.append(np.nan)
                            else:
                                desc_vals.append(float(value))
                        except Exception:
                            desc_vals.append(np.nan)
                        desc_names.append(f'RDKit_RDKit_{desc_name}')
                    
                    if desc_names:
                        desc_array = np.array(desc_vals, dtype=float)
                        all_arrays.append(desc_array)
                        all_names += desc_names
                
                # Combine all features for this molecule
                if all_arrays:
                    combined = np.concatenate(all_arrays).astype(float)
                    all_results.append(combined)
                else:
                    # No features needed
                    all_results.append(np.array([]))
                    
            except Exception as e:
                processing_errors += 1
                # For failed molecules, add empty array
                all_results.append(np.array([]))
        
        # Create final DataFrame
        if all_results and len(all_results[0]) > 0:
            # Get column names from first successful molecule
            first_mol = valid_mols[0]
            temp_names = []
            
            if needs_morgan:
                temp_names += [f'RDKit_Morgan_{i}' for i in range(morgan_size)]
            if needs_morgan6:
                temp_names += [f'RDKit_Morgan6_{i}' for i in range(morgan_size)]
            if needs_morgancount:
                temp_names += [f'RDKit_MorganCount_{i}' for i in range(morgan_size)]
            if needs_torsion:
                temp_names += [f'RDKit_Torsion_{i}' for i in range(torsion_size)]
            if needs_avalon:
                temp_names += [f'RDKit_Avalon_{i}' for i in range(avalon_size)]
            if needs_rdkfp:
                temp_names += [f'RDKit_RDKitFP_{i}' for i in range(rdkitfp_size)]
            if needs_scaffold:
                temp_names += [f'RDKit_Scaffold_{i}' for i in range(scaffold_size)]
            if needs_descriptors:
                for desc_name, fn in Descriptors.descList:
                    if needed_desc_names is None or desc_name in needed_desc_names:
                        temp_names.append(f'RDKit_RDKit_{desc_name}')
            
            # Create DataFrame with proper column names
            rdkit_df = pd.DataFrame(all_results, columns=temp_names)
            
            # Validate the dataframe
            if rdkit_df.shape[0] != len(valid_mols):
                print(f"Warning: RDKit output has {rdkit_df.shape[0]} rows, expected {len(valid_mols)}")
            
            # Reorder to match original SMILES order, filling missing with NaN
            result_df = pd.DataFrame(index=range(len(smiles_list)), columns=rdkit_df.columns)
            for orig_idx, batch_idx in enumerate(valid_indices):
                if batch_idx < len(rdkit_df):
                    result_df.iloc[orig_idx] = rdkit_df.iloc[batch_idx]
            
            print(f"Successfully generated {len(result_df.columns)} RDKit descriptors for {len(valid_mols)}/{len(smiles_list)} molecules")
            if processing_errors > 0:
                print(f"RDKit processing errors for {processing_errors} molecules")
            
            return result_df
        else:
            print("No RDKit features were generated")
            return None
            
    except Exception as e:
        print(f"Error in RDKit batch processing: {e}")
        return None


# --- Helper Function for Model Diagnostics ---
def diagnose_model_features(model, model_type):
    """Diagnose model feature requirements"""
    feature_info = {
        'feature_names': None,
        'expected_count': None,
        'source': 'unknown'
    }
    
    # Try different ways to get feature information
    if hasattr(model, 'feature_names_in_'):
        feature_info['feature_names'] = [str(name) for name in model.feature_names_in_]
        feature_info['expected_count'] = len(model.feature_names_in_)
        feature_info['source'] = 'feature_names_in_'
    elif hasattr(model, 'feature_names_'):
        feature_info['feature_names'] = [str(name) for name in model.feature_names_]
        feature_info['expected_count'] = len(model.feature_names_)
        feature_info['source'] = 'feature_names_'
    elif hasattr(model, 'booster') and model_type == "xgboost":
        try:
            feature_names = model.booster().feature_names
            if feature_names:
                feature_info['feature_names'] = [str(name) for name in feature_names]
                feature_info['expected_count'] = len(feature_names)
                feature_info['source'] = 'xgboost_booster'
        except:
            pass
    
    return feature_info


# --- Enhanced Model Loading Function ---
@st.cache_resource
def load_model(model_path):
    """Load different types of models with enhanced type detection for future flexibility"""
    file_extension = os.path.splitext(model_path)[1].lower()
    
    if file_extension == ".pkl":
        # Load pickle models (sklearn, xgboost, etc.)
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = joblib.load(model_path)
            # Surface sklearn version mismatch as a gentle sidebar notice
            for w in caught_warnings:
                if issubclass(w.category, UserWarning) and 'InconsistentVersion' in str(w.message):
                    pass  # surfaced at call site via st.sidebar
            
            # Enhanced model type detection based on class name and module
            model_class_name = model.__class__.__name__.lower()
            model_module = model.__class__.__module__.lower()
            
            # XGBoost detection (multiple variants)
            if any(keyword in model_class_name for keyword in ["xgb", "xgboost", "xgregressor", "xgclassifier"]):
                return model, "xgboost"
            elif any(keyword in model_module for keyword in ["xgboost", "xgb"]):
                return model, "xgboost"
                
            # LightGBM detection
            elif any(keyword in model_class_name for keyword in ["lgb", "lightgbm", "lgbregressor", "lgbclassifier"]):
                return model, "lightgbm"
            elif any(keyword in model_module for keyword in ["lightgbm", "lgb"]):
                return model, "lightgbm"
                
            # CatBoost detection
            elif any(keyword in model_class_name for keyword in ["catboost", "catregressor", "catclassifier"]):
                return model, "catboost"
            elif any(keyword in model_module for keyword in ["catboost"]):
                return model, "catboost"
                
            # AdaBoost and Gradient Boosting (sklearn-based)
            elif any(keyword in model_class_name for keyword in ["adaboost", "gradientboosting", "histgradientboosting"]):
                return model, "sklearn"
                
            # Random Forest and Tree-based models
            elif any(keyword in model_class_name for keyword in ["randomforest", "extratrees", "decisiontree"]):
                return model, "sklearn"
                
            # SVM models
            elif any(keyword in model_class_name for keyword in ["svc", "svr", "svm"]):
                return model, "sklearn"
                
            # Linear models
            elif any(keyword in model_class_name for keyword in ["linear", "logistic", "ridge", "lasso", "elastic"]):
                return model, "sklearn"
                
            # Neural Network models (sklearn)
            elif any(keyword in model_class_name for keyword in ["mlpregressor", "mlpclassifier"]):
                return model, "sklearn"
                
            # Ensemble methods
            elif any(keyword in model_class_name for keyword in ["voting", "bagging", "stacking"]):
                return model, "sklearn"
                
            # Default to sklearn for other models
            else:
                return model, "sklearn"
                
        except Exception as e:
            # Try with pickle as fallback
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                # Default to sklearn for pickle-loaded models
                return model, "sklearn"
            except Exception as e2:
                raise ValueError(f"Could not load pickle model: {e}, {e2}")
    
    elif file_extension == ".cbm":
        # CatBoost native model
        if not CATBOOST_AVAILABLE:
            raise ValueError("CatBoost not available. Install catboost to load .cbm models")
        try:
            model = cb.CatBoostRegressor()
            model.load_model(model_path)
            return model, "catboost"
        except Exception as e:
            raise ValueError(f"Could not load CatBoost model: {e}")
    
    elif file_extension in [".joblib", ".dump"]:
        try:
            model = joblib.load(model_path)
            model_class_name = model.__class__.__name__.lower()
            if any(k in model_class_name for k in ["catboost", "catregressor"]):
                return model, "catboost"
            return model, "sklearn"
        except Exception as e:
            raise ValueError(f"Could not load joblib model: {e}")
    
    else:
        raise ValueError(f"Unsupported model file extension: {file_extension}. Supported: .pkl, .cbm, .joblib")



# --- Enhanced Prediction Function with Single Molecule Processing ---
def _is_valid_numeric_prediction(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

def predict_from_smiles_progressive(smiles_list, names_list, model, model_type, selected_descriptor_names, progress_placeholder=None, scaler=None, header_timer_placeholder=None, backup_model=None, backup_model_type=None, backup_selected_descriptor_names=None, disagreement_threshold=0.5):
    """Process molecules one at a time with independent timer and smooth progress reporting"""
    total_molecules = len(smiles_list)
    results = []
    
    # Use provided placeholder or create new ones
    if progress_placeholder is None:
        progress_placeholder = st.empty()
    
    # Use the global timer - it should already be running
    global_timer = st.session_state.global_timer
    
    # Enhanced time tracking for estimation only (not for display timer)
    molecule_times = []  # Store individual processing times
    molecule_start_time = time.time()  # Track start for estimates
    
    with progress_placeholder.container():
        # Initialize progress tracking within the placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_container = st.empty()
        results_table_container = st.empty()  # Container for growing results table
    
    for idx, smiles in enumerate(smiles_list):
        name = names_list[idx] if names_list else f"Molecule_{idx+1}"
        molecule_start_time = time.time()
        
        # Update progress
        progress = (idx + 1) / total_molecules
        progress_bar.progress(progress)
        
        # Show overall progress (no per-molecule details)
        status_text.info(f"🔬 Processing molecules... ({idx + 1}/{total_molecules})")
        
        # Process single molecule with scaler (time tracking continues uninterrupted)
        molecule_result = process_single_molecule(smiles, name, model, model_type, selected_descriptor_names, idx == 0, scaler)
        diag_ad = molecule_result[4] if len(molecule_result) > 4 else "N/A"

        if backup_model is not None and backup_model_type is not None:
            backup_result = process_single_molecule(
                smiles, name, backup_model, backup_model_type, backup_selected_descriptor_names, False, scaler
            )

            primary_ok = _is_valid_numeric_prediction(molecule_result[2])
            backup_ok = _is_valid_numeric_prediction(backup_result[2])

            if primary_ok:
                primary_pic50 = float(molecule_result[2])
                primary_ic50 = float(molecule_result[3])
            else:
                primary_pic50 = molecule_result[2]
                primary_ic50 = molecule_result[3]

            if backup_ok:
                backup_pic50 = float(backup_result[2])
                backup_ic50 = float(backup_result[3])
            else:
                backup_pic50 = backup_result[2]
                backup_ic50 = backup_result[3]

            if primary_ok and backup_ok:
                disagreement = abs(primary_pic50 - backup_pic50)
                if disagreement <= disagreement_threshold:
                    consensus_flag = "Agree"
                else:
                    consensus_flag = "Disagree"
                final_pic50 = primary_pic50
                final_ic50 = primary_ic50
                molecule_result = [
                    name,
                    smiles,
                    round(primary_pic50, 3),
                    round(primary_ic50, 2),
                    round(backup_pic50, 3),
                    round(backup_ic50, 2),
                    round(disagreement, 3),
                    consensus_flag,
                    round(final_pic50, 3),
                    round(final_ic50, 2),
                    diag_ad
                ]
            elif primary_ok and not backup_ok:
                molecule_result = [
                    name,
                    smiles,
                    round(primary_pic50, 3),
                    round(primary_ic50, 2),
                    backup_pic50,
                    backup_ic50,
                    "N/A",
                    "Backup Error",
                    round(primary_pic50, 3),
                    round(primary_ic50, 2),
                    diag_ad
                ]
            elif (not primary_ok) and backup_ok:
                molecule_result = [
                    name,
                    smiles,
                    primary_pic50,
                    primary_ic50,
                    round(backup_pic50, 3),
                    round(backup_ic50, 2),
                    "N/A",
                    "Primary Error - Backup Used",
                    round(backup_pic50, 3),
                    round(backup_ic50, 2),
                    diag_ad
                ]
            else:
                molecule_result = [
                    name,
                    smiles,
                    primary_pic50,
                    primary_ic50,
                    backup_pic50,
                    backup_ic50,
                    "N/A",
                    "Both Error",
                    "Error",
                    "Error",
                    diag_ad
                ]

        results.append(molecule_result)
        
        # Record processing time for this molecule (for estimation purposes only)
        molecule_end_time = time.time()
        molecule_processing_time = molecule_end_time - molecule_start_time
        molecule_times.append(molecule_processing_time)
        molecule_start_time = molecule_end_time  # Reset for next molecule
        
        # Get independent timer display (always continuous)
        elapsed_total_time = global_timer.get_elapsed_time()
        elapsed_str = global_timer.format_time(elapsed_total_time)
        
        # Update header timer if placeholder is provided
        if header_timer_placeholder:
            header_timer_placeholder.markdown(f"### ⏱️ {elapsed_str}")
        
        # Calculate remaining time with adaptive estimation (for progress display only)
        if len(molecule_times) >= 1:
            # Use recent processing times for better estimation
            recent_times = molecule_times[-min(3, len(molecule_times)):]  # Last 3 or fewer
            avg_time_per_molecule = sum(recent_times) / len(recent_times)
            remaining_molecules = total_molecules - (idx + 1)
            estimated_remaining_time = avg_time_per_molecule * remaining_molecules
            
            # Format remaining time display in HH:MM:SS format
            remaining_hours = int(estimated_remaining_time // 3600)
            remaining_minutes = int((estimated_remaining_time % 3600) // 60)
            remaining_seconds = int(estimated_remaining_time % 60)
            remaining_str = f"{remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
            
            # Show only remaining time estimate (header shows used time)
            time_container.info(f"⏳ Estimated remaining: ~{remaining_str}")
        else:
            # First molecule - calculating estimate
            time_container.info(f"⏳ Calculating time estimate...")
        
        # Live updates removed - results shown at end
    
    # Final cleanup - show completion message and results
    progress_bar.progress(1.0)
    
    # Get final time from independent timer
    total_time = global_timer.get_elapsed_time()
    final_time_str = global_timer.format_time(total_time)
    
    # Calculate average time per molecule for final stats
    avg_per_molecule = total_time / total_molecules if total_molecules > 0 else 0
    
    status_text.success(f"✅ Processing complete! {total_molecules} molecules in {final_time_str} (avg: {avg_per_molecule:.1f}s per molecule)")
    time_container.empty()
    
    # Show final results table
    update_live_results_table(results, results_table_container, total_molecules)
    
    # Stop the timer
    global_timer.stop()
    
    return results

def predict_from_smiles_batch(smiles_list, names_list, model, model_type, selected_descriptor_names, progress_placeholder=None, scaler=None, header_timer_placeholder=None, backup_model=None, backup_model_type=None, backup_selected_descriptor_names=None, disagreement_threshold=0.5):
    """Process all molecules in parallel and return results at once - optimized with batch PaDEL processing"""
    total_molecules = len(smiles_list)
    results = []
    
    # Use provided placeholder or create new ones
    if progress_placeholder is None:
        progress_placeholder = st.empty()
    
    # Use the global timer
    global_timer = st.session_state.global_timer
    
    with progress_placeholder.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_container = st.empty()
    
    # Start timer
    global_timer.start()
    
    status_text.info(f"🔬 Processing {total_molecules} molecules in parallel...")
    
    # Determine which features to calculate based on model requirements
    needs_padel = any(f.startswith('PaDEL_') for f in (selected_descriptor_names or []))
    needs_mordred = any(f.startswith('Mordred_') for f in (selected_descriptor_names or []))
    needs_rdkit = any(f.startswith('RDKit_') for f in (selected_descriptor_names or []))
    
    # Determine which features to calculate
    # If scaler is present, calculate ALL features it expects, otherwise only model features
    scaler_feature_names = list(scaler.feature_names_in_) if (scaler and hasattr(scaler, 'feature_names_in_')) else None
    
    if scaler_feature_names:
        # Calculate ALL features that the scaler expects
        needs_padel_all = any(f.startswith('PaDEL_') for f in scaler_feature_names)
        needs_mordred_all = any(f.startswith('Mordred_') for f in scaler_feature_names)  
        needs_rdkit_all = any(f.startswith('RDKit_') for f in scaler_feature_names)
        
        # Override the model-based selection to include all scaler features
        needs_padel = needs_padel or needs_padel_all
        needs_mordred = needs_mordred or needs_mordred_all
        needs_rdkit = needs_rdkit or needs_rdkit_all
        
        # For RDKit, calculate all features if scaler needs any RDKit features
        rdkit_required = None if needs_rdkit_all else selected_descriptor_names
    else:
        # No scaler, use model-based feature selection
        rdkit_required = selected_descriptor_names
    
    # Batch process all descriptors if needed (much more efficient)
    padel_batch_df = None
    mordred_batch_df = None
    rdkit_batch_df = None
    
    if needs_padel:
        status_text.info(f"🧪 Calculating PaDEL descriptors for {total_molecules} molecules...")
        padel_batch_df = get_padel_features_batch(smiles_list)
        if padel_batch_df is not None:
            status_text.info(f"✅ PaDEL descriptors calculated for {len(padel_batch_df)} molecules")
        else:
            status_text.warning("⚠️ PaDEL descriptor calculation failed, skipping PaDEL features")
    
    if needs_mordred:
        status_text.info(f"🧪 Calculating Mordred descriptors for {total_molecules} molecules...")
        mordred_batch_df = get_mordred_features_batch(smiles_list)
        if mordred_batch_df is not None:
            status_text.info(f"✅ Mordred descriptors calculated for {len(mordred_batch_df)} molecules")
        else:
            status_text.warning("⚠️ Mordred descriptor calculation failed, skipping Mordred features")
    
    if needs_rdkit:
        status_text.info(f"🧪 Calculating RDKit descriptors for {total_molecules} molecules...")
        # Generate all RDKit features if scaler requires specific features, otherwise only selected ones
        rdkit_required = None if (scaler and hasattr(scaler, 'feature_names_in_')) else selected_descriptor_names
        rdkit_batch_df = get_rdkit_features_batch(smiles_list, rdkit_required)
        if rdkit_batch_df is not None:
            status_text.info(f"✅ RDKit descriptors calculated for {len(rdkit_batch_df)} molecules")
        else:
            status_text.warning("⚠️ RDKit descriptor calculation failed, skipping RDKit features")
    
    # Process all molecules in parallel using threads
    # Use a simple approach that maintains context properly
    max_workers = min(4, total_molecules)  # Limit to 4 threads
    
    # Prepare arguments
    args_list = [(smiles_list[idx], names_list[idx] if names_list else f"Molecule_{idx+1}", 
                  model, model_type, selected_descriptor_names, scaler, scaler_feature_names,
                  padel_batch_df.iloc[idx] if padel_batch_df is not None and idx < len(padel_batch_df) else None,
                  mordred_batch_df.iloc[idx] if mordred_batch_df is not None and idx < len(mordred_batch_df) else None,
                  rdkit_batch_df.iloc[idx] if rdkit_batch_df is not None and idx < len(rdkit_batch_df) else None,
                  backup_model, backup_model_type, backup_selected_descriptor_names, disagreement_threshold) 
                 for idx in range(len(smiles_list))]
    
    # Use thread pool for remaining processing with warning suppression for ScriptRunContext
    # (threads don't interact with Streamlit UI, only perform computation)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_molecule_batch, *args) for args in args_list]
            
            # Collect results
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=120)  # 120 second timeout per molecule
                results.append(result)
                progress = (i + 1) / total_molecules
                progress_bar.progress(progress)
            except Exception as e:
                error_msg = f"Timeout or error" if "timeout" in str(e).lower() else f"Parallel error: {str(e)}"
                results.append([f"Molecule_{i+1}", smiles_list[i], error_msg, "Error", "N/A"])
    
    # Final updates
    progress_bar.progress(1.0)
    total_time = global_timer.get_elapsed_time()
    final_time_str = global_timer.format_time(total_time)
    avg_per_molecule = total_time / total_molecules if total_molecules > 0 else 0
    
    status_text.success(f"✅ Processing complete! {total_molecules} molecules in {final_time_str} (avg: {avg_per_molecule:.1f}s per molecule)")
    time_container.empty()
    
    # Display final results table
    if results:
        use_backup_columns = len(results[0]) >= 11 if results else False
        if use_backup_columns:
            trimmed_results = [row[:11] for row in results]
            result_df = pd.DataFrame(trimmed_results, columns=["Name", "SMILES", "Primary pIC50", "Primary IC50 (nM)", "Backup pIC50", "Backup IC50 (nM)", "ΔpIC50", "Consensus", "Final pIC50", "Final IC50 (nM)", "AD (Max Tanimoto)"])
            prediction_col = "Final pIC50"
        else:
            trimmed_results = [row[:5] for row in results]
            result_df = pd.DataFrame(trimmed_results, columns=["Name", "SMILES", "pIC50", "IC50 (nM)", "AD (Max Tanimoto)"])
            prediction_col = "pIC50"

        result_df = result_df.drop(columns=["Scaled", "Coverage", "Scaler Status", "Feature Coverage"], errors='ignore')
        
        # Count successful predictions
        error_types = ['Error', 'Prediction Error', 'Invalid Prediction', 'Feature Mismatch']
        successful_preds = len(result_df[~result_df[prediction_col].isin(error_types)])
        
        # Display with color coding
        def highlight_errors(val):
            if val in error_types:
                return 'background-color: #ffebee; color: #c62828; font-weight: bold;'
            return ''
        
        styled_df = result_df.style.map(highlight_errors)
        
        st.markdown(f"### 📊 Final Results ({total_molecules} molecules | {successful_preds} successful)")
        st.dataframe(styled_df, width='stretch')
        
        # Download results
        st.markdown("---")
        csv_data = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv_data,
            file_name=f"qsar_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    global_timer.stop()
    
    return results

def process_single_molecule_batch(smiles, name, model, model_type, selected_descriptor_names, scaler, scaler_feature_names, padel_row=None, mordred_row=None, rdkit_row=None, backup_model=None, backup_model_type=None, backup_selected_descriptor_names=None, disagreement_threshold=0.5):
    """Process a single molecule for batch execution - optimized with pre-calculated descriptor data"""
    try:
        model_target_features = selected_descriptor_names
        
        # Generate features - use pre-calculated batch data where available
        needs_padel = any(f.startswith('PaDEL_') for f in (selected_descriptor_names or []))
        needs_mordred = any(f.startswith('Mordred_') for f in (selected_descriptor_names or []))
        needs_rdkit = any(f.startswith('RDKit_') for f in (selected_descriptor_names or []))
        
        # Use pre-calculated data or calculate individually
        descriptor_quality_all = {
            'rdkit': {'morgan': {'calculated': False, 'failed': False}},
            'padel': {'calculated': False, 'failed': False},
            'mordred': {'calculated': False, 'failed': False}
        }
        
        if needs_padel and padel_row is not None:
            padel_df = pd.DataFrame([padel_row.values], columns=padel_row.index)
        else:
            padel_result = get_padel_features(smiles) if needs_padel else (None, {})
            if isinstance(padel_result, tuple):
                padel_df, padel_quality = padel_result
                if 'padel' in padel_quality:
                    descriptor_quality_all['padel'] = padel_quality['padel']
            else:
                padel_df = padel_result
            
        if needs_mordred and mordred_row is not None:
            mordred_df = pd.DataFrame([mordred_row.values], columns=mordred_row.index)
        else:
            mordred_result = get_mordred_features(smiles) if needs_mordred else (None, {})
            if isinstance(mordred_result, tuple):
                mordred_df, mordred_quality = mordred_result
                if 'mordred' in mordred_quality:
                    descriptor_quality_all['mordred'] = mordred_quality['mordred']
            else:
                mordred_df = mordred_result
            
        if needs_rdkit and rdkit_row is not None:
            rdkit_df = pd.DataFrame([rdkit_row.values], columns=rdkit_row.index)
        else:
            rdkit_result = get_rdkit_features(smiles, selected_descriptor_names) if needs_rdkit else (None, {})
            if isinstance(rdkit_result, tuple):
                rdkit_df, rdkit_quality = rdkit_result
                descriptor_quality_all['rdkit'] = rdkit_quality
            else:
                rdkit_df = rdkit_result
                rdkit_quality = {}
        
        # Calculate AD AFTER descriptors are combined (using pre-calculated data)
        if ENABLE_AD_CHECK:
            training_padel_df, training_mordred_df, training_rdkit_df = load_training_descriptors_cached()
            ad_summary = calculate_ad_tanimoto(
                padel_row, mordred_row, rdkit_row, 
                selected_descriptor_names,
                training_padel_df, training_mordred_df, training_rdkit_df
            )
        else:
            ad_summary = "N/A (disabled)"
        
        feature_dfs = [df for df in [padel_df, mordred_df, rdkit_df] if df is not None]
        if not feature_dfs:
            return [name, smiles, "No features generated", "Error", ad_summary]
        
        features = pd.concat(feature_dfs, axis=1)
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        features.columns = features.columns.astype(str)
        
        # Step 1: Apply scaler to ALL scaler features (if scaler is present)
        if scaler and scaler_feature_names:
            # Create a DataFrame with ALL scaler features in the correct order
            # Fill with calculated values where available, NaN otherwise
            scaler_features_df = pd.DataFrame(index=features.index, columns=scaler_feature_names)
            
            # Fill with calculated features
            for col in features.columns:
                if col in scaler_features_df.columns:
                    scaler_features_df[col] = features[col]
            
            # Fill missing features with NaN (to indicate they couldn't be calculated)
            # The scaler and model should handle NaN appropriately
            scaler_features_df = scaler_features_df.fillna(np.nan)
            
            # Apply scaler - it should handle NaN if the underlying data had NaN during training
            scaled_values = _safe_transform(scaler, scaler_features_df)
            scaled_df = pd.DataFrame(scaled_values, columns=scaler_feature_names, index=features.index)
            scaler_status = f"Scaled (all {len(scaler_feature_names)})"
        else:
            scaled_df = features
            scaler_status = "Raw"
        
        # Step 2: Apply model feature priority - subset to only model features for prediction
        if model_target_features:
            missing_model = [feat for feat in model_target_features if feat not in scaled_df.columns]
            if missing_model:
                return [name, smiles, f"Missing model features: {len(missing_model)}", "Error", ad_summary]
            
            X_for_prediction_df = scaled_df.loc[:, model_target_features].copy()
            feature_coverage = f"{X_for_prediction_df.shape[1]}/{len(model_target_features)}"
        else:
            X_for_prediction_df = scaled_df
            feature_coverage = f"{X_for_prediction_df.shape[1]}/{scaled_df.shape[1]}"
        
        # Predict with primary model (pass DataFrame to preserve feature names)
        if model_type == "catboost":
            pred = model.predict(X_for_prediction_df)[0]
        else:
            pred = model.predict(X_for_prediction_df)[0]
        
        pIC50_val = float(pred)
        IC50_nM = 10 ** (9 - pIC50_val)
        
        # Check if backup model is available and should be used
        if backup_model is not None and backup_model_type is not None and backup_selected_descriptor_names is not None:
            try:
                # Prepare backup model features
                backup_model_target_features = backup_selected_descriptor_names
                
                # Subset to backup model features
                if backup_model_target_features:
                    missing_backup = [feat for feat in backup_model_target_features if feat not in scaled_df.columns]
                    if missing_backup:
                        # If backup features are missing, skip backup prediction
                        backup_pic50 = "Backup features missing"
                        backup_ic50 = "N/A"
                        disagreement = "N/A"
                        consensus_flag = "No backup"
                        final_pic50 = pIC50_val
                        final_ic50 = IC50_nM
                    else:
                        X_backup_df = scaled_df.loc[:, backup_model_target_features].copy()
                        
                        # Predict with backup model (pass DataFrame to preserve feature names)
                        if backup_model_type == "catboost":
                            backup_pred = backup_model.predict(X_backup_df)[0]
                        else:
                            backup_pred = backup_model.predict(X_backup_df)[0]
                        
                        backup_pic50_val = float(backup_pred)
                        backup_ic50_val = 10 ** (9 - backup_pic50_val)
                        
                        backup_pic50 = round(backup_pic50_val, 3)
                        backup_ic50 = round(backup_ic50_val, 2)
                        
                        # Calculate disagreement and consensus
                        disagreement_val = abs(pIC50_val - backup_pic50_val)
                        disagreement = round(disagreement_val, 3)
                        
                        if disagreement_val <= disagreement_threshold:
                            consensus_flag = "Agree"
                        else:
                            consensus_flag = "Disagree"
                        
                        final_pic50 = pIC50_val  # Use primary model as final
                        final_ic50 = IC50_nM
                else:
                    # No backup features specified
                    backup_pic50 = "No backup features"
                    backup_ic50 = "N/A"
                    disagreement = "N/A"
                    consensus_flag = "No backup"
                    final_pic50 = pIC50_val
                    final_ic50 = IC50_nM
                    
            except Exception as backup_e:
                # Backup prediction failed
                backup_pic50 = f"Backup error: {str(backup_e)[:30]}"
                backup_ic50 = "N/A"
                disagreement = "N/A"
                consensus_flag = "Backup error"
                final_pic50 = pIC50_val
                final_ic50 = IC50_nM
        else:
            # No backup model available
            backup_pic50 = "No backup"
            backup_ic50 = "N/A"
            disagreement = "N/A"
            consensus_flag = "No backup"
            final_pic50 = pIC50_val
            final_ic50 = IC50_nM
        
        # Return extended format with backup information
        return [name, smiles, round(pIC50_val, 3), round(IC50_nM, 2), backup_pic50, backup_ic50, disagreement, consensus_flag, round(final_pic50, 3), round(final_ic50, 2), ad_summary]
    
    except Exception as e:
        # For errors, return appropriate format based on whether backup is enabled
        if backup_model is not None and backup_model_type is not None:
            # Return extended format for consistency
            return [name, smiles, f"Error: {str(e)[:30]}", "Error", "N/A", "N/A", "N/A", "Error", f"Error: {str(e)[:30]}", "N/A", "N/A"]
        else:
            # Return basic format
            return [name, smiles, f"Error: {str(e)[:50]}", "Error", "N/A"]

def update_live_results_table(results, results_container, total_molecules):
    """Update the live results table as each molecule is processed"""
    if not results:
        return
        
    use_backup_columns = len(results[0]) >= 11
    if use_backup_columns:
        trimmed_results = [row[:11] for row in results]
        result_df = pd.DataFrame(trimmed_results, columns=["Name", "SMILES", "Primary pIC50", "Primary IC50 (nM)", "Backup pIC50", "Backup IC50 (nM)", "ΔpIC50", "Consensus", "Final pIC50", "Final IC50 (nM)", "AD (Max Tanimoto)"])
        prediction_col = "Final pIC50"
    else:
        trimmed_results = [row[:5] for row in results]
        result_df = pd.DataFrame(trimmed_results, columns=["Name", "SMILES", "pIC50", "IC50 (nM)", "AD (Max Tanimoto)"])
        prediction_col = "pIC50"

    result_df = result_df.drop(columns=["Scaled", "Coverage", "Scaler Status", "Feature Coverage"], errors='ignore')
    
    # Count successful predictions so far (for additional info)
    error_types = ['Error', 'Prediction Error', 'Invalid Prediction', 'Feature Mismatch']
    successful_preds = len(result_df[~result_df[prediction_col].isin(error_types)])
    
    # Display growing results with color coding
    def highlight_errors(val):
        if val in error_types:
            return 'background-color: #ffebee; color: #c62828; font-weight: bold;'
        return ''
    
    styled_df = result_df.style.map(highlight_errors)
    
    with results_container.container():
        # Show processed/total instead of successful/total
        st.markdown(f"### 📊 Live Results (Total molecules = {total_molecules} | Processed • {successful_preds} successful)")
        
        # Always show all results as they accumulate
        st.dataframe(styled_df, width='stretch')

def map_model_features_to_scaler(model_features, scaler_features):
    """Map model feature names (without prefixes) to scaler feature names (with prefixes)"""
    feature_mapping = {}
    scaler_feature_set = set(scaler_features)
    
    # All prefix namespaces used during training (order = preference when ambiguous)
    ALL_PREFIXES = ['PaDEL_', 'Mordred_', 'RDKit_']

    def strip_prefixes(feature_name):
        base_name = str(feature_name)
        changed = True
        while changed:
            changed = False
            for prefix in ALL_PREFIXES:
                if base_name.startswith(prefix):
                    base_name = base_name[len(prefix):]
                    changed = True
        return base_name

    scaler_base_lookup = {}
    for scaler_feat in scaler_features:
        scaler_base_lookup.setdefault(strip_prefixes(scaler_feat), []).append(scaler_feat)

    def canonicalize_feature_name(feature_name):
        return ''.join(ch.lower() for ch in str(feature_name) if ch.isalnum())

    scaler_canonical_lookup = {}
    for scaler_feat in scaler_features:
        scaler_canonical_lookup.setdefault(canonicalize_feature_name(scaler_feat), []).append(scaler_feat)
        scaler_canonical_lookup.setdefault(canonicalize_feature_name(strip_prefixes(scaler_feat)), []).append(scaler_feat)

    for model_feat in model_features:
        # Try direct match first
        if model_feat in scaler_feature_set:
            feature_mapping[model_feat] = model_feat
            continue
            
        # Try with common prefixes
        possible_matches = []
        for prefix in ALL_PREFIXES:
            candidate = f"{prefix}{model_feat}"
            if candidate in scaler_feature_set:
                possible_matches.append(candidate)
        
        if len(possible_matches) == 1:
            feature_mapping[model_feat] = possible_matches[0]
        elif len(possible_matches) > 1:
            # Multiple matches - prefer PaDEL, then Mordred, then RDKit
            for preferred_prefix in ALL_PREFIXES:
                preferred_match = f"{preferred_prefix}{model_feat}"
                if preferred_match in possible_matches:
                    feature_mapping[model_feat] = preferred_match
                    break
        else:
            model_base = strip_prefixes(model_feat)
            base_matches = scaler_base_lookup.get(model_base, [])
            if len(base_matches) == 1:
                feature_mapping[model_feat] = base_matches[0]
            elif len(base_matches) > 1:
                selected = None
                for preferred_prefix in ALL_PREFIXES:
                    candidate = next((m for m in base_matches if m.startswith(preferred_prefix)), None)
                    if candidate is not None:
                        selected = candidate
                        break
                if selected is not None:
                    feature_mapping[model_feat] = selected

            if model_feat not in feature_mapping:
                canonical_matches = scaler_canonical_lookup.get(canonicalize_feature_name(model_feat), [])
                if not canonical_matches:
                    canonical_matches = scaler_canonical_lookup.get(canonicalize_feature_name(model_base), [])
                if len(canonical_matches) == 1:
                    feature_mapping[model_feat] = canonical_matches[0]
                elif len(canonical_matches) > 1:
                    selected = None
                    for preferred_prefix in ALL_PREFIXES:
                        candidate = next((m for m in canonical_matches if m.startswith(preferred_prefix)), None)
                        if candidate is not None:
                            selected = candidate
                            break
                    if selected is not None:
                        feature_mapping[model_feat] = selected
    
    return feature_mapping

def map_target_features_to_generated_columns(target_features, available_columns):
    """Map target feature names to generated descriptor columns, tolerating training prefixes."""
    feature_mapping = {}
    available_feature_set = set(available_columns)
    all_prefixes = ['PaDEL_', 'Mordred_', 'RDKit_']

    def strip_prefixes(feature_name):
        base_name = str(feature_name)
        changed = True
        while changed:
            changed = False
            for prefix in all_prefixes:
                if base_name.startswith(prefix):
                    base_name = base_name[len(prefix):]
                    changed = True
        return base_name

    available_base_lookup = {}
    for available_feature in available_columns:
        available_base_lookup.setdefault(strip_prefixes(available_feature), []).append(available_feature)

    def canonicalize_feature_name(feature_name):
        return ''.join(ch.lower() for ch in str(feature_name) if ch.isalnum())

    available_canonical_lookup = {}
    for available_feature in available_columns:
        available_canonical_lookup.setdefault(canonicalize_feature_name(available_feature), []).append(available_feature)
        available_canonical_lookup.setdefault(canonicalize_feature_name(strip_prefixes(available_feature)), []).append(available_feature)

    for target_feature in target_features:
        if target_feature in available_feature_set:
            feature_mapping[target_feature] = target_feature
            continue

        for prefix in all_prefixes:
            prefixed_feature = f"{prefix}{target_feature}"
            if prefixed_feature in available_feature_set:
                feature_mapping[target_feature] = prefixed_feature
                break

        if target_feature in feature_mapping:
            continue

        target_base = strip_prefixes(target_feature)
        base_matches = available_base_lookup.get(target_base, [])
        if len(base_matches) == 1:
            feature_mapping[target_feature] = base_matches[0]
        elif len(base_matches) > 1:
            selected = None
            for preferred_prefix in all_prefixes:
                candidate = next((m for m in base_matches if m.startswith(preferred_prefix)), None)
                if candidate is not None:
                    selected = candidate
                    break
            if selected is None:
                selected = base_matches[0]
            feature_mapping[target_feature] = selected

        if target_feature not in feature_mapping:
            canonical_matches = available_canonical_lookup.get(canonicalize_feature_name(target_feature), [])
            if not canonical_matches:
                canonical_matches = available_canonical_lookup.get(canonicalize_feature_name(target_base), [])
            if len(canonical_matches) == 1:
                feature_mapping[target_feature] = canonical_matches[0]
            elif len(canonical_matches) > 1:
                selected = None
                for preferred_prefix in all_prefixes:
                    candidate = next((m for m in canonical_matches if m.startswith(preferred_prefix)), None)
                    if candidate is not None:
                        selected = candidate
                        break
                if selected is None:
                    selected = canonical_matches[0]
                feature_mapping[target_feature] = selected

    return feature_mapping

def process_single_molecule(smiles, name, model, model_type, selected_descriptor_names, show_descriptor_info=False, scaler=None):
    """Process a single molecule and return prediction result"""
    try:
        scaler_status = "Raw"
        feature_coverage = "N/A"
        ad_summary = "N/A"  # Will be calculated after descriptors are computed

        model_target_features = None
        if selected_descriptor_names is not None and len(selected_descriptor_names) > 0:
            model_target_features = [str(n) for n in selected_descriptor_names]

        scaler_feature_names = None
        if scaler is not None and hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            scaler_feature_names = [str(n) for n in scaler.feature_names_in_]

        # Generate features (parallelized). If scaler schema is available,
        # generation MUST satisfy scaler-required features first (before model subsetting).
        # Otherwise fall back to model-targeted generation.
        generation_target_features = scaler_feature_names if scaler_feature_names else model_target_features

        needs_padel = True
        needs_mordred = True
        needs_rdkit = True
        if generation_target_features is not None and len(generation_target_features) > 0:
            needs_padel = any(f.startswith('PaDEL_') for f in generation_target_features)
            needs_mordred = any(f.startswith('Mordred_') for f in generation_target_features)
            needs_rdkit = any(f.startswith('RDKit_') for f in generation_target_features)

        padel_df = None
        mordred_df = None
        rdkit_df = None
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_mordred = executor.submit(get_mordred_features, smiles) if (MORDRED_AVAILABLE and needs_mordred) else None
                rdkit_required = None
                if generation_target_features is not None and len(generation_target_features) > 0:
                    rdkit_required = [f for f in generation_target_features if str(f).startswith('RDKit_')]
                fut_rdkit = executor.submit(get_rdkit_features, smiles, rdkit_required) if needs_rdkit else None
                fut_padel = executor.submit(get_padel_features, smiles) if (PADEL_AVAILABLE and needs_padel) else None

                mordred_result = fut_mordred.result() if fut_mordred is not None else None
                
                # Extract quality metrics from all descriptor results
                rdkit_quality = {}
                mordred_quality = {}
                padel_quality = {}
                
                rdkit_result = fut_rdkit.result() if fut_rdkit is not None else None
                if isinstance(rdkit_result, tuple):
                    rdkit_df, rdkit_quality = rdkit_result
                else:
                    rdkit_df = rdkit_result
                
                # Handle mordred tuple result
                if isinstance(mordred_result, tuple):
                    mordred_df, mordred_quality = mordred_result
                else:
                    mordred_df = mordred_result
                
                padel_result = fut_padel.result() if fut_padel is not None else None
                if isinstance(padel_result, tuple):
                    padel_df, padel_quality = padel_result
                else:
                    padel_df = padel_result

        if needs_rdkit and rdkit_df is None:
            return [name, smiles, "Error - Invalid SMILES", "Error", ad_summary]

        # Calculate AD NOW using pre-calculated descriptors (single-row versions)
        if ENABLE_AD_CHECK:
            padel_row = padel_df.iloc[0] if padel_df is not None and len(padel_df) > 0 else None
            mordred_row = mordred_df.iloc[0] if mordred_df is not None and len(mordred_df) > 0 else None
            rdkit_row = rdkit_df.iloc[0] if rdkit_df is not None and len(rdkit_df) > 0 else None
            
            training_padel_df, training_mordred_df, training_rdkit_df = load_training_descriptors_cached()
            ad_summary = calculate_ad_tanimoto(
                padel_row, mordred_row, rdkit_row,
                selected_descriptor_names,
                training_padel_df, training_mordred_df, training_rdkit_df
            )
        else:
            ad_summary = "N/A (disabled)"

        # Combine all available descriptor types
        feature_dfs = []
        feature_counts = []
        
        if padel_df is not None:
            feature_dfs.append(padel_df)
            feature_counts.append(f"PaDEL: {len(padel_df.columns)}")
        
        if mordred_df is not None:
            feature_dfs.append(mordred_df)
            feature_counts.append(f"Mordred: {len(mordred_df.columns)}")
        
        if rdkit_df is not None:
            feature_dfs.append(rdkit_df)
            feature_counts.append(f"RDKit: {len(rdkit_df.columns)}")
        
        if not feature_dfs:
            return [name, smiles, "Error - No descriptors available", "Error", ad_summary]
        
        features = pd.concat(feature_dfs, axis=1)
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        features.columns = features.columns.astype(str)
        
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0.0)

        # ---------------------------------------------------------------
        # Enhanced inference pipeline with robust feature alignment:
        # 1) Align generated descriptors to scaler training feature schema (with fallbacks)
        # 2) Apply scaler transform in the exact training feature order
        # 3) Keep only model-trained features (with fallbacks)
        # ---------------------------------------------------------------
        scaler_aligned_df = None
        scaled_df = None

        # Step 1: align to scaler training schema (strict matching)
        if scaler_feature_names is not None:
            missing_scaler = [feat for feat in scaler_feature_names if feat not in numeric_features.columns]
            if missing_scaler:
                missing_sample = sorted(set(missing_scaler))[:10]
                raise ValueError(
                    f"Missing {len(set(missing_scaler))} features from scaler training data: {missing_sample}"
                    + (" ..." if len(set(missing_scaler)) > 10 else "")
                )

            scaler_aligned_df = numeric_features.loc[:, scaler_feature_names].copy()
            scaler_aligned_df.columns = scaler_aligned_df.columns.astype(str)
            scaler_status = "Aligned"
        else:
            scaler_aligned_df = numeric_features
            scaler_status = "Raw"

        # Step 2: apply scaling in training order
        if scaler is not None:
            try:
                scaled_values = _safe_transform(scaler, scaler_aligned_df)
                scaled_df = pd.DataFrame(scaled_values, columns=scaler_aligned_df.columns, index=scaler_aligned_df.index)
                scaler_status += " + Scaled"
                if show_descriptor_info:
                    st.success("✅ Applied training scaler")
            except Exception as e:
                if show_descriptor_info:
                    st.warning(f"⚠️ Scaler transformation failed: {e}")
                scaled_df = scaler_aligned_df
                scaler_status += " (scaler failed)"
        else:
            scaled_df = scaler_aligned_df
            if show_descriptor_info:
                st.info("ℹ️ No scaler applied - using raw features")

        # Step 3: keep only features used to train the selected model (strict matching)
        if model_target_features is not None and len(model_target_features) > 0:
            exact_model_features = [feat for feat in model_target_features if feat in scaled_df.columns]
            missing_model = [feat for feat in model_target_features if feat not in scaled_df.columns]

            # Backward-compatible fallback only when model features are unprefixed.
            # This keeps strict behavior for already-prefixed training features.
            if missing_model and all('_' not in m for m in missing_model):
                fallback_mapping = map_target_features_to_generated_columns(missing_model, scaled_df.columns)
                for target_feat, src_feat in fallback_mapping.items():
                    if target_feat in missing_model:
                        exact_model_features.append(src_feat)
                missing_model = [feat for feat in missing_model if feat not in fallback_mapping]

            if missing_model:
                missing_sample = sorted(missing_model)[:10]
                raise ValueError(
                    f"Model expects {len(missing_model)} unavailable features: {missing_sample}"
                    + (" ..." if len(missing_model) > 10 else "")
                )

            if len(exact_model_features) == len(model_target_features):
                # Exact ordered subset (best case)
                X_for_prediction_df = scaled_df.loc[:, model_target_features].copy()
            else:
                # Fallback path where some model features were unprefixed and mapped.
                # Preserve target order.
                aligned_cols = []
                for feat in model_target_features:
                    if feat in scaled_df.columns:
                        aligned_cols.append(feat)
                    else:
                        mapped = map_target_features_to_generated_columns([feat], scaled_df.columns).get(feat)
                        aligned_cols.append(mapped)
                X_for_prediction_df = scaled_df.loc[:, aligned_cols].copy()
                X_for_prediction_df.columns = model_target_features

            feature_coverage = f"{X_for_prediction_df.shape[1]}/{len(model_target_features)}"
        else:
            X_for_prediction_df = scaled_df
            feature_coverage = f"{X_for_prediction_df.shape[1]}/{X_for_prediction_df.shape[1]}"

        X_for_prediction_df.columns = X_for_prediction_df.columns.astype(str)
        X_for_prediction_df = X_for_prediction_df.reset_index(drop=True)

        # Handle feature count mismatch for models without explicit feature names
        expected_features = get_expected_feature_count(model, model_type)
        if expected_features and expected_features != X_for_prediction_df.shape[1]:
            if expected_features < X_for_prediction_df.shape[1]:
                X_for_prediction_df = X_for_prediction_df.iloc[:, :expected_features]
            else:
                return [name, smiles, "Feature Mismatch", "Feature Mismatch", ad_summary]
        
        # Make prediction based on model type (pass DataFrame to preserve feature names)
        if model_type == "sklearn":
            pred = model.predict(X_for_prediction_df)[0]
        
        elif model_type == "xgboost":
            try:
                # XGBRegressor can work with DataFrames or numpy arrays
                pred = model.predict(X_for_prediction_df)[0]
            except Exception as e:
                if show_descriptor_info:
                    st.error(f"XGBoost prediction failed: {str(e)}")
                return [name, smiles, f"XGBoost Error: {str(e)[:30]}", "Error", ad_summary]
        
        elif model_type == "lightgbm":
            pred = model.predict(X_for_prediction_df)[0]
        
        elif model_type == "catboost":
            pred = model.predict(X_for_prediction_df)[0]
        
        else:
            return [name, smiles, "Unsupported Model Type", "Unsupported Model Type", ad_summary]
        
        # Convert to IC50
        pIC50_val = float(pred)
        IC50_nM = 10 ** (9 - pIC50_val)
        
        return [name, smiles, round(pIC50_val, 3), round(IC50_nM, 2), ad_summary]
    
    except Exception as e:
        return [name, smiles, f"Error: {str(e)[:50]}", "Error", "N/A"]

def get_expected_feature_count(model, model_type):
    """Get expected feature count from model"""
    try:
        if hasattr(model, 'n_features_in_'):
            return model.n_features_in_
        return None
    except:
        return None

def _safe_transform(scaler, X):
    """Apply scaler transform strictly using the loaded scaler."""
    return scaler.transform(X)


# --- Load Scaler ---
@st.cache_resource
def load_scaler():
    """Load the scaler used during training"""
    scaler_path = (
        find_existing_file(("scaling", "train_scaler.pkl"))
        or find_existing_file(("app", "scaling", "train_scaler.pkl"))
    )

    if not scaler_path:
        st.warning("⚠️ No scaler found in scaling folder")
        return None

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            scaler = joblib.load(scaler_path)

        st.session_state['loaded_scaler_path'] = scaler_path

        # Check for sklearn version mismatch warning
        version_warnings = [
            w for w in caught_warnings
            if issubclass(w.category, UserWarning) and 'InconsistentVersion' in str(w.message)
        ]
        if version_warnings:
            msg = str(version_warnings[0].message)
            # Extract versions if present
            import re
            m = re.search(r'version (\S+) when using version (\S+)', msg)
            if m:
                st.sidebar.warning(
                    f"⚠️ Scaler was saved with sklearn {m.group(1)} but running {m.group(2)}. "
                    f"Predictions may be affected. Re-save the scaler with the current sklearn version to remove this warning."
                )
            else:
                st.sidebar.warning("⚠️ Scaler sklearn version mismatch — consider re-saving with current version.")
        return scaler
    except Exception as e:
        st.warning(f"⚠️ Could not load scaler: {e}")
        return None

# --- Load Available Models ---
@st.cache_data
def get_available_models():
    """Get all available model files with their types"""
    models_dir = find_existing_dir(("models",)) or find_existing_dir(("app", "models"))

    if not models_dir:
        return []
    
    model_files = []
    supported_extensions = {
        ".pkl": "Scikit-learn/Joblib/Boosting",
        ".joblib": "Scikit-learn/Joblib",
        ".dump": "Joblib Dump",
        ".cbm": "CatBoost",
    }
    
    for file in os.listdir(models_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in supported_extensions:
            model_files.append({
                "filename": file,
                "type": supported_extensions[ext],
                "path": os.path.join(models_dir, file)
            })
    
    return model_files

# Display model selection with enhanced UI
available_models = get_available_models()
selected_model_file = None
selected_model_info = None
enable_backup_model = False
backup_model_file = None
backup_model_info = None

if not available_models:
    st.error("❌ No models found in the 'models' directory!")
    st.info("📁 **Supported formats:**")
    st.info("• **Supported**: .pkl/.joblib/.dump (sklearn/CatBoost), .cbm (CatBoost native)")
else:
    # Create a nice display of available models
    st.sidebar.markdown("### 🔧 Model Selection")
    
    model_options = []
    for model in available_models:
        display_name = f"{model['filename']} ({model['type']})"
        model_options.append(display_name)

    # Default selection priority: validated CatBoost -> CatBoost native -> any CatBoost -> first available
    default_model_index = 0
    preferred_exact = [
        "catboost_optimized_model_validated.cbm",
        "catboost_optimized_model.cbm",
        "catboost_optimized_model.pkl",
    ]

    # First pass: exact preferred filenames
    for preferred_name in preferred_exact:
        hit_index = next(
            (i for i, m in enumerate(available_models) if m["filename"].lower() == preferred_name),
            None,
        )
        if hit_index is not None:
            default_model_index = hit_index
            break

    # Second pass fallback: any CatBoost model
    if default_model_index == 0 and not any(
        m["filename"].lower() == preferred_exact[0] for m in available_models
    ):
        catboost_index = next(
            (
                i
                for i, m in enumerate(available_models)
                if "catboost" in m["filename"].lower() or m["type"].lower() == "catboost"
            ),
            None,
        )
        if catboost_index is not None:
            default_model_index = catboost_index
    
    selected_display = st.sidebar.selectbox("Select Model", model_options, index=default_model_index)
    
    # Find the selected model
    selected_model_file = None
    selected_model_info = None
    for i, option in enumerate(model_options):
        if option == selected_display:
            selected_model_file = available_models[i]["filename"] 
            selected_model_info = available_models[i]
            break

    extra_trees_candidates = [
        model for model in available_models
        if ("extratrees" in model["filename"].lower()) and (model["filename"] != selected_model_file)
    ]

    if extra_trees_candidates:
        enable_backup_model = st.sidebar.checkbox("Use ExtraTrees backup model", value=True)
        if enable_backup_model:
            backup_options = [f"{model['filename']} ({model['type']})" for model in extra_trees_candidates]
            backup_selected_display = st.sidebar.selectbox("Backup Model", backup_options)
            for i, option in enumerate(backup_options):
                if option == backup_selected_display:
                    backup_model_file = extra_trees_candidates[i]["filename"]
                    backup_model_info = extra_trees_candidates[i]
                    break

if selected_model_file:
    model_path = selected_model_info["path"]
    
    # Display model information
    st.sidebar.markdown(f"**Selected Model:** {selected_model_file}")
    st.sidebar.markdown(f"**Type:** {selected_model_info['type']}")
    
    try:
        with st.spinner(f"🔄 Loading {selected_model_info['type']} model..."):
            model, model_type = load_model(model_path)
        
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.markdown(f"**Detected Type:** {model_type}")
        
        # Load scaler used during training
        scaler = load_scaler()
        if scaler is not None:
            st.sidebar.success(f"✅ Training scaler loaded")
            scaler_type_name = type(scaler).__name__
            if hasattr(scaler, 'feature_range'):
                st.sidebar.markdown(f"**Scaler:** {scaler_type_name} {scaler.feature_range}")
            else:
                st.sidebar.markdown(f"**Scaler:** {scaler_type_name}")

            # Verify the scaler actually works; offer repair if broken
            try:
                if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                    _test = pd.DataFrame(
                        np.zeros((1, scaler.n_features_in_)),
                        columns=[str(c) for c in scaler.feature_names_in_]
                    )
                else:
                    _test = np.zeros((1, scaler.n_features_in_))
                _safe_transform(scaler, _test)
                scaler_ok = True
            except Exception:
                scaler_ok = False

            if not scaler_ok:
                st.sidebar.error("❌ Scaler is broken (version mismatch). Click below to repair.")

            if st.sidebar.button("🔧 Repair & Resave Scaler", help="Resave the scaler with the current sklearn version to fix version-mismatch errors"):
                try:
                    import sklearn
                    scaler_path = (
                        find_existing_file(("scaling", "train_scaler.pkl"))
                        or find_existing_file(("app", "scaling", "train_scaler.pkl"))
                    )
                    if not scaler_path:
                        raise FileNotFoundError("Could not locate train_scaler.pkl for repair")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _scaler_raw = joblib.load(scaler_path)
                    joblib.dump(_scaler_raw, scaler_path)
                    st.sidebar.success(f"✅ Scaler resaved with sklearn {sklearn.__version__}. Please reload the page.")
                    load_scaler.clear()  # clear cache so it reloads
                    st.rerun()
                except Exception as repair_err:
                    st.sidebar.error(f"❌ Repair failed: {repair_err}")
        else:
            st.sidebar.warning("⚠️ No scaler found - using raw features")
        
        # Diagnose model features
        feature_info = diagnose_model_features(model, model_type)
        selected_descriptor_names = feature_info['feature_names']
        
        if selected_descriptor_names:
            pass
        elif feature_info['expected_count']:
            selected_descriptor_names = None
        else:
            st.sidebar.warning("⚠️ Model feature requirements unknown")
            st.sidebar.info("📊 Will attempt to use all available molecular descriptors")
            selected_descriptor_names = None

        backup_model = None
        backup_model_type = None
        backup_selected_descriptor_names = None
        disagreement_threshold = 0.5

        if enable_backup_model and backup_model_file:
            backup_model_path = backup_model_info["path"]
            try:
                backup_model, backup_model_type = load_model(backup_model_path)
                backup_feature_info = diagnose_model_features(backup_model, backup_model_type)
                backup_selected_descriptor_names = backup_feature_info['feature_names']
                st.sidebar.success(f"✅ Backup model loaded: {backup_model_file}")
                disagreement_threshold = st.sidebar.number_input(
                    "Disagreement threshold (ΔpIC50)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                )
            except Exception as backup_error:
                st.sidebar.error(f"❌ Backup model load failed: {str(backup_error)}")
                backup_model = None
                backup_model_type = None
                backup_selected_descriptor_names = None
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        scaler = None  # Ensure scaler is defined even if loading fails
        st.stop()


    # Display usage limits
    col_info1, col_info2 = st.columns([2, 1])
    with col_info1:
            # Enhanced File Upload UI
        st.markdown("### 📂 Upload Molecular Data")
        st.markdown("Upload a `.csv` or `.txt` file containing molecule name and SMILES strings")
        st.markdown("""
        **Example SMILES format:**
        ```
        CHEMBL97 CCOc1nn(-c2cccc(OCc3ccccc3)c2)c(=O)o1 
        ```
        ---
        """)
    with col_info2:
        st.info(f"📊 **Processing Limit**: Maximum {MAX_MOLECULES_LIMIT} molecules per session")
        st.info(f"⚡ **Recommended**: Process {RECOMMENDED_BATCH_SIZE} molecules or fewer for optimal performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose file", 
            type=["csv", "txt"],
            help="CSV files should have 'SMILES' column (and optional 'Name' column). TXT files should have SMILES or 'Name SMILES' per line."
        )
    
    with col2:
        # Sample file download
        sample_df = pd.DataFrame({
            'Name': ["Ethanol", "Acetic_Acid", "Benzene", "Triethylamine", "Aspirin", "Caffeine"],
            'SMILES': ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
        })
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Sample File",
            data=sample_csv,
            file_name="sample_molecules.csv",
            mime="text/csv",
            help="Download a sample CSV file to see the expected format"
        )
        if st.button("📋 Use Example Data"):
            # Create example data
            example_smiles = [
                "CCO",  # Ethanol
                "CC(=O)O",  # Acetic acid
                "c1ccccc1",  # Benzene
                "CCN(CC)CC"  # Triethylamine
            ]
            example_names = ["Ethanol", "Acetic_Acid", "Benzene", "Triethylamine"]
            
            # Create a temporary DataFrame to simulate file upload
            example_df = pd.DataFrame({
                'Name': example_names,
                'SMILES': example_smiles
            })
            
            st.session_state['example_data'] = example_df
            st.info("📋 Example data loaded!")
            
    # Handle file upload or example data
    smiles_list, names_list = [], []
    data_source = None
    
    if uploaded_file:
        data_source = "uploaded_file"
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if 'SMILES' in df.columns:
                smiles_list = df['SMILES'].tolist()
                names_list = df['Name'].tolist() if 'Name' in df.columns else [f"Molecule_{i+1}" for i in range(len(df))]
            else:
                st.error("❌ CSV must contain a 'SMILES' column.")
        elif uploaded_file.name.endswith(".txt"):
            lines = uploaded_file.read().decode("utf-8").splitlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) == 2:
                    names_list.append(parts[0])
                    smiles_list.append(parts[1])
                elif len(parts) == 1:
                    smiles_list.append(parts[0])
                    names_list.append(f"Molecule_{i+1}")
    
    elif 'example_data' in st.session_state:
        data_source = "example"
        df = st.session_state['example_data']
        smiles_list = df['SMILES'].tolist()
        names_list = df['Name'].tolist()
        
        # Show example data info with clear button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📋 Example data loaded: {len(smiles_list)} molecules")
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_example_data"):
                del st.session_state['example_data']
                st.rerun()

    if smiles_list:
        # Enforce molecule limit to prevent mining/abuse
        if len(smiles_list) > MAX_MOLECULES_LIMIT:
            st.error(f"❌ **Upload Limit Exceeded**: Your file contains {len(smiles_list)} molecules, but the maximum allowed is {MAX_MOLECULES_LIMIT} molecules per session.")
            st.warning(f"🔄 **To proceed**: Please split your data into smaller batches of {MAX_MOLECULES_LIMIT} molecules or fewer.")
            st.info(f"💡 **Tip**: For processing large datasets, consider running multiple smaller batches to avoid system overload and ensure reliable results.")
            
            # Show a preview of the data but don't process
            preview_df = pd.DataFrame({
                'Name': names_list[:10] if names_list else [f"Molecule_{i+1}" for i in range(min(10, len(smiles_list)))],
                'SMILES': smiles_list[:10]
            })
            st.markdown("**Data Preview (first 10 molecules):**")
            st.dataframe(preview_df, width='stretch')
            st.stop()  # Stop execution here
        
        # Show warning for large batches (but still allow processing)
        elif len(smiles_list) > RECOMMENDED_BATCH_SIZE:
            st.warning(f"⚠️ **Large Batch Warning**: Processing {len(smiles_list)} molecules. For optimal performance, consider processing {RECOMMENDED_BATCH_SIZE} molecules or fewer at a time.")
            st.info("💭 You can continue, but processing may take longer and use more computational resources.")
        
        # Start the independent timer as soon as data is available
        global_timer = st.session_state.global_timer
        global_timer.start()
        
        # Create columns for heading and timer display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🧪 Processing {len(smiles_list)} molecules...")
        
        with col2:
            # Create a placeholder for the live timer
            header_timer_placeholder = st.empty()
            # Initial timer display
            elapsed_str = global_timer.format_time(global_timer.get_elapsed_time())
            header_timer_placeholder.markdown(f"### ⏱️ {elapsed_str}")
        
        # Show progress and run prediction with real-time updates
        st.info("🔬 Starting molecular descriptor calculation and prediction...")
        
        # Create a placeholder for progress tracking only
        progress_placeholder = st.empty()
        
        # Use the batch prediction function for parallel processing
        results = predict_from_smiles_batch(
            smiles_list, names_list, model, model_type, selected_descriptor_names, 
            progress_placeholder, scaler, header_timer_placeholder,
            backup_model=backup_model,
            backup_model_type=backup_model_type,
            backup_selected_descriptor_names=backup_selected_descriptor_names,
            disagreement_threshold=disagreement_threshold
        )

        # Keep the results table visible and just show completion message
        use_backup_columns = len(results[0]) >= 11 if results else False
        if use_backup_columns:
            result_df = pd.DataFrame(results, columns=["Name", "SMILES", "Primary pIC50", "Primary IC50 (nM)", "Backup pIC50", "Backup IC50 (nM)", "ΔpIC50", "Consensus", "Final pIC50", "Final IC50 (nM)", "AD (Max Tanimoto)"])
            metric_column = "Final pIC50"
        else:
            result_df = pd.DataFrame(results, columns=["Name", "SMILES", "pIC50", "IC50 (nM)", "AD (Max Tanimoto)"])
            metric_column = "pIC50"
        error_types = ['Error', 'Prediction Error', 'Invalid Prediction', 'Feature Mismatch']
        successful_preds = len(result_df[~result_df[metric_column].isin(error_types)])

        # --- Auto-save to logs/ folder ---
        try:
            import hashlib
            source_name = uploaded_file.name if uploaded_file else "example"
            clean_name = os.path.splitext(source_name)[0].replace(" ", "_")[:20]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_hash = hashlib.md5(f"{timestamp}{clean_name}".encode()).hexdigest()[:6].upper()
            job_id = f"public_{timestamp[:8]}_{clean_name}-{job_hash}"

            job_dir = os.path.join(LOGS_DIR, job_id)
            os.makedirs(job_dir, exist_ok=True)

            input_df = pd.DataFrame({"Name": names_list, "SMILES": smiles_list})
            input_df.to_csv(os.path.join(job_dir, f"{job_id}_input.csv"), index=False)
            result_df.to_csv(os.path.join(job_dir, f"{job_id}_results.csv"), index=False)

            elapsed_seconds = round(global_timer.get_elapsed_time(), 1)
            log_data = {
                "job_id": job_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": "public",
                "model_file": selected_model_file,
                "model_type": model_type,
                "total_molecules": len(smiles_list),
                "successful_predictions": successful_preds,
                "failed_predictions": len(smiles_list) - successful_preds,
                "elapsed_time_seconds": elapsed_seconds
            }
            with open(os.path.join(job_dir, f"{job_id}_log.json"), "w") as f:
                json.dump(log_data, f, indent=2)

            jobs_log_path = os.path.join(LOGS_DIR, "jobs.log")
            log_line = f"{log_data['timestamp']} | {job_id} | public | {selected_model_file} | {successful_preds}/{len(smiles_list)} molecules | {elapsed_seconds}s\n"
            with open(jobs_log_path, "a") as f:
                f.write(log_line)

            st.success(f"✅ Results auto-saved to {job_dir}")
        except Exception as _save_err:
            st.warning(f"⚠️ Auto-save failed: {_save_err}")

        if successful_preds == 0:
            st.error("❌ No successful predictions. Please check your SMILES strings.")
        
        # Final summary and download
        st.markdown("### 📥 Download Results")
        
        # Final results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Molecules", len(smiles_list))
        with col2:
            st.metric("Successful Predictions", successful_preds)
        with col3:
            if successful_preds > 0:
                # Filter out error values and convert to numeric
                valid_mask = ~result_df[metric_column].isin(error_types)
                valid_preds = pd.to_numeric(result_df.loc[valid_mask, metric_column], errors='coerce')
                valid_preds = valid_preds.dropna()  # Remove any NaN values
                
                if len(valid_preds) > 0:
                    avg_pic50 = valid_preds.mean()
                    st.metric("Average pIC50", f"{avg_pic50:.2f}")
                else:
                    st.metric("Average pIC50", "N/A")
        
        # Download button
        if successful_preds > 0:
            csv = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f'<a href="data:file/csv;base64,{b64}" download="ic50_predictions.csv" class="download-btn">📥 Download Results (CSV)</a>', 
                unsafe_allow_html=True
            )

# --- Enhanced Footer ---
st.markdown("---")
st.markdown(f"""
<div class="footer">
<p><strong>🏛️ KEAP1 IC50 Predictor - Texas Christian University</strong></p>
<p><strong>⚖️ Usage Limits:</strong> Max {MAX_MOLECULES_LIMIT} molecules per session • Recommended batch size: {RECOMMENDED_BATCH_SIZE} molecules</p>
<p><strong>Developed by:</strong> Department of Chemistry and Biochemistry, Texas Christian University</p>
<p><em>Advancing molecular property prediction through innovative machine learning approaches</em></p>
</div>
""", unsafe_allow_html=True)