import numpy as np
from flare.io import otf_parser
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp.sparse_gp import SGP_Wrapper
import warnings

try:
    from flare.bffs.sgp._C_flare import SparseGP, NormalizedDotProduct, B2, DotProduct
except ImportError as e:
    Warning.warn(f"Could not import _C_flare: {e.__class__.__name__}: {e}.")

# load lammps-otf_flare.json
model_dir = './'
sgp_file = f'{model_dir}/lammps-otf_flare.json'

import json
# set up ASE flare calculator
with open(sgp_file, 'r') as f:
    gp_dct = json.loads(f.readline())
    if gp_dct.get("class", None) == "SGP_Calculator":
        flare_calc, kernels = SGP_Calculator.from_file(sgp_file)
    else:
        sgp, kernels = SGP_Wrapper.from_file(sgp_file)
        flare_calc = SGP_Calculator(sgp)


from flare.scripts.otf_train import get_super_cell, get_dft_calc
import yaml

# config the dft calculator and supercell
config_file = f'{model_dir}/lammps.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
    otf_config = config["otf"]
super_cell = get_super_cell(config["supercell"])
dft_calc = get_dft_calc(config["dft_calc"])

# predict the energy and force fro structures, 使用方法与ASE相同
# if we use flare calculator, we must wrap it in OTF, use FLARE_Atoms

from flare.scripts.otf_train import OTF
from ase.calculators.lammpsrun import LAMMPS

otf = OTF(atoms=super_cell, flare_calc=flare_calc, dft_calc=dft_calc, **otf_config)

flare_calc.calculate(atoms=otf.atoms)
flare_calc_e = flare_calc.results.get('energy', None)
flare_calc_f = flare_calc.results.get('forces', None)
flare_calc_s = flare_calc.results.get('stress', None)

flare_calc_e_1 = flare_calc.results['energy']
flare_calc_f_1 = flare_calc.results['forces']

# 在OTF中 atoms 有一个属性，flare results，
# 这个属性是一个字典，里面存储了energy和forces
flare_atoms_e = otf.atoms.energy
print(f'flare_calc_e: {flare_calc_e}')      # -332.18692436639685
print(f'flare_calc_f: {flare_calc_f}')      # [[ -3.07878154e-01 -1.00920507e-01  2.26388432e-01] ...]
print(f'flare_calc_s: {flare_calc_s}')      # [-1.16636430e-03 -1.22592367e-03 -3.44829727e-03 -9.08749919e-06 -1.05114523e-04 -3.70648447e-05]
print(f'flare_calc_e_1: {flare_calc_e_1}')      # -332.18692436639685
print(f'flare_calc_f_1: {flare_calc_f_1}')      # [[ -3.07878154e-01 -1.00920507e-01  2.26388432e-01] ...]
print(f'flare_atoms_e: {flare_atoms_e}')        # -332.18692436639685
