import yaml
import json

from ase.io import iread
from flare.atoms import FLARE_Atoms
from flare.scripts.otf_train import OTF
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp.sparse_gp import SGP_Wrapper
from flare.scripts.otf_train import get_super_cell, get_dft_calc
try:
    from flare.bffs.sgp._C_flare import SparseGP, NormalizedDotProduct, B2, DotProduct
except ImportError as e:
    Warning.warn(f"Could not import _C_flare: {e.__class__.__name__}: {e}.")


# ---------in this script, we need sgp_model.json and otf_dft.yaml ------------
# ---------you can get the result from the training process --------------------------


# load lammps-otf_flare.json
model_dir = './boron_offline_820'
sgp_file = f'{model_dir}/offline_820_flare.json'
config_file = f'{model_dir}/offline.yaml'
dataset = f'{model_dir}/training_set.xyz'
# --------------------------------------------------------------------------------

# config the dft calculator
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    otf_config = config["otf"]


# load the sgp model, set up ASE flare calculator
def load_sgp(sgp_file):
    with open(sgp_file, 'r') as f:
        gp_dct = json.loads(f.readline())
        if gp_dct.get("class", None) == "SGP_Calculator":
            flare_calc, kernels = SGP_Calculator.from_file(sgp_file)
        else:
            sgp, kernels = SGP_Wrapper.from_file(sgp_file)
            flare_calc = SGP_Calculator(sgp)
        return flare_calc, kernels

# construct the OTF object
def get_otf(otf_config):
    flare_calc, kernels = load_sgp(sgp_file)
    flare_atoms = get_super_cell(config["supercell"])
    dft_calc = get_dft_calc(config["dft_calc"])
    otf = OTF(atoms=flare_atoms, flare_calc=flare_calc, dft_calc=dft_calc, **otf_config)
    return


def predict(otf, dft_calc, flare_calc):
    flare_calc.calculate(atoms=otf.atoms)
    flare_calc_e = flare_calc.results.get('energy', None)
    flare_calc_f = flare_calc.results.get('forces', None)
    flare_calc_s = flare_calc.results.get('stress', None)

    dft_calc.calculate(atoms=otf.atoms)
    dft_calc_e = dft_calc.results.get('energy', None)
    dft_calc_f = dft_calc.results.get('forces', None)
    dft_calc_s = dft_calc.results.get('stress', None)
    return flare_calc_e, flare_calc_f, flare_calc_s, dft_calc_e, dft_calc_f, dft_calc_s

ase_traj = iread(dataset, index=':', format='extxyz')
print(f'len(ase_traj): {len(list(ase_traj))}')
for atoms in iread(dataset, index=':', format='extxyz')[:1]:
    print(f'atoms: {atoms}')
    flare_atoms = FLARE_Atoms.from_ase_atoms(atoms)
    otf.atoms = flare_atoms    
    num_atoms = len(otf.atoms)
    print(f'num_atoms: {num_atoms}')

    flare_calc_e, flare_calc_f, flare_calc_s, flare_calc_std, dft_calc_e, dft_calc_f, dft_calc_s, dft_calc_std = predict(otf, dft_calc, flare_calc)
   
    print(f'energy/atoms: {flare_calc_e/num_atoms}')  
    print(f'forces[:3]: {flare_calc_f[:3]}')
    print(f'stress: {flare_calc_s}')

    
    print(f'energy/atoms: {dft_calc_e/num_atoms}')
    print(f'forces[:3]: {dft_calc_f[:3]}')
    print(f'stress: {dft_calc_s}')
