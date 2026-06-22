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