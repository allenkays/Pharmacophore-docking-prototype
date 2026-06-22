"""
Pharmacophore docking prototype - my attempt at a quick rigid aligner using RDKit.
I pieced this together after fiddling with conformers and features for a while.
Workflow roughly:
1. Load targets JSON
2. Build mols + conformers
3. Extract features
4. Try triple alignments + score
etc.
Requirements: pipenv install numpy scipy rdkit
"""

# imports - kept most of them together but added a comment because why not
import itertools
import json
import numpy as np
import os
from dataclasses import dataclass
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Config stuff - tweak these if things are too slow or missing poses
# ---------------------------------------------------------------------------
INPUT_JSON = "/root/data/targets.json"
OUTPUT_SDF = "/root/results/docked_poses.sdf"

NUM_CONFORMERS = 300   # quite a few, but helps sampling
RANDOM_SEED = 42
PRUNE_RMS_THRESHOLD = 0.5
MAX_MMFF_ITERATIONS = 2000

EXCLUSION_RADIUS = 1.2
CLASH_TOLERANCE = 0.1
PHARMACOPHORE_WIDTH = 1.25
MAX_ALIGNMENT_TRIPLES = 4000  # cap this or it explodes on big mols

# RDKit feature factory setup - this was a pain to get right initially
FEATURE_DEFINITION_FILE = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(FEATURE_DEFINITION_FILE)

# map families because RDKit sometimes returns LumpedHydrophobe etc.
FAMILY_MAP = {
    "Donor": "Donor",
    "Acceptor": "Acceptor",
    "Hydrophobe": "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "Aromatic": "Aromatic",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
# Simple data holders
@dataclass
class PharmacophoreSite:
    """One desired interaction point in the target pharmacophore."""
    family: str
    position: np.ndarray  # I prefer numpy arrays here
    weight: float

@dataclass
class DockingResult:
    """Result for one molecule - best pose info."""
    mol: Chem.Mol
    coordinates: np.ndarray
    score: float
    conformer_energy: float
    conformer_id: int

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def load_targets(json_path: str):
    """Load targets from JSON. Basic validation because bad data happens."""
    with open(json_path, "r", encoding="utf-8") as f:  # shorter var name
        targets = json.load(f)
    
    if not isinstance(targets, list):
        raise ValueError("Expected a list in targets.json")
    
    # quick sanity check
    for i, t in enumerate(targets):
        required = {"smiles", "interaction_sites", "excluded_volumes"}
        missing = required - set(t.keys())
        if missing:
            raise ValueError(f"Target #{i} missing keys: {missing}")
    return target

def parse_sites(target):
    """Turn JSON sites into our objects. Skipped type hints here to save time."""
    sites = []
    for s in target.get("interaction_sites", []):
        fam = s["family"]
        if fam not in list(FAMILY_MAP.values()):
            raise ValueError(f"Bad family {fam} - check your JSON")
        
        pos = np.array([s["x"], s["y"], s["z"]], dtype=float)
        sites.append(PharmacophoreSite(family=fam, position=pos, weight=float(s["weight"])))
    return sites

def parse_exclusion_centers(target):
    """Exclusion volumes -> numpy array."""
    vols = target.get("excluded_volumes", [])
    if not vols:
        return np.empty((0, 3), dtype=float)
    return np.array([[v["x"], v["y"], v["z"]] for v in vols], dtype=float)

# ---------------------------------------------------------------------------
# Molecule builder from SMILES - wrapped in a function for clarity
# ---------------------------------------------------------------------------
def build_molecule(smiles):
    """SMILES -> RDKit mol with Hs. Pretty standard."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Couldn't parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)  # need hydrogens for features usually
    return mol

def generate_and_optimize_conformers(mol, num_conformers=NUM_CONFORMERS):
    """Generate + MMFF optimize. Returns ids and energies."""
    conf_ids = list(AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_conformers,
        randomSeed=RANDOM_SEED,
        pruneRmsThresh=PRUNE_RMS_THRESHOLD,
    ))
    if not conf_ids:
        raise RuntimeError("No conformers generated :(")
    
    opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=MAX_MMFF_ITERATIONS)
    energies = {}
    for cid, (status, en) in zip(conf_ids, opt_results):
        energies[cid] = float(en)  # keep even if not fully converged
    return conf_ids, energies

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def get_features_by_family(mol, conformer_id):
    """Pull pharmacophore features for a specific conformer."""
    features_by_family = {
        "Donor": [], "Acceptor": [], "Hydrophobe": [], "Aromatic": []
    }
    
    rdkit_feats = FEATURE_FACTORY.GetFeaturesForMol(mol, confId=conformer_id)
    for f in rdkit_feats:
        fam = FAMILY_MAP.get(f.GetFamily())
        if fam:
            features_by_family[fam].append(list(f.GetPos()))
    
    # convert to numpy
    return {
        fam: (np.array(pos, dtype=float) if pos else np.empty((0, 3), dtype=float))
        for fam, pos in features_by_family.items()
    }

def flatten_features(features_by_family):
    """Turn grouped features into a flat list for triple selection."""
    flat = []
    for fam, positions in features_by_family.items():
        for p in positions:
            flat.append({"family": fam, "position": p})
    return flat

# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------
def is_degenerate(points, tolerance=1e-3):
    """Check if points are too few or collinear for a good transform."""
    if len(points) < 3:
        return True
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] < 1e-8:
        return True
    return (singular_values[1] / singular_values[0]) < tolerance

def fit_rigid_transform(moving_points, target_points):
    """Find rotation + translation."""
    m_center = moving_points.mean(axis=0)
    t_center = target_points.mean(axis=0)
    rot = Rotation.align_vectors(target_points - t_center, moving_points - m_center)[0].as_matrix()
    trans = t_center - rot @ m_center
    return rot, trans

def apply_transform(coords, rot, trans):
    """Apply rigid transform to coordinates."""
    return coords @ rot.T + trans

# ---------------------------------------------------------------------------
# Triple generation
# ---------------------------------------------------------------------------
def candidate_triples(sites, flat_features, max_triples=MAX_ALIGNMENT_TRIPLES):
    """Generate compatible three-point correspondences."""
    # precompute compatible features per site
    compat = []
    for site in sites:
        idxs = [i for i, f in enumerate(flat_features) if f["family"] == site.family]
        compat.append(idxs)
    
    yielded = 0
    for site_idx in itertools.combinations(range(len(sites)), 3):
        i0, i1, i2 = site_idx
        if not (compat[i0] and compat[i1] and compat[i2]):
            continue
        for fidx in itertools.product(compat[i0], compat[i1], compat[i2]):
            if len(set(fidx)) < 3:
                continue
            sel_sites = [sites[i0], sites[i1], sites[i2]]
            sel_feats = [flat_features[k] for k in fidx]
            yield sel_sites, sel_feats
            yielded += 1
            if yielded >= max_triples:
                return
