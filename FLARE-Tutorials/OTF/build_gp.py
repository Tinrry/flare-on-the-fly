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


# predict the energy and force fro structures, 使用方法与ASE相同
# if we use flare calculator, we must wrap it in OTF, use FLARE_Atoms

from flare.scripts.otf_train import OTF

dft_calc = get_dft_calc(config["dft_calc"])
# wrap the flare Atoms
otf = OTF(atoms=super_cell, flare_calc=flare_calc, dft_calc=dft_calc, **otf_config)

flare_calc.calculate(atoms=otf.atoms)
flare_calc_e = flare_calc.results.get('energy', None)
flare_calc_f = flare_calc.results.get('forces', None)
flare_calc_s = flare_calc.results.get('stress', None)
flare_calc_std = flare_calc.results.get('stds', None)
print(f'flare_calc_e: {flare_calc_e}')      

dft_calc.calculate(atoms=otf.atoms)
dft_calc_e = dft_calc.results.get('energy', None)
dft_calc_f = dft_calc.results.get('forces', None)
dft_calc_s = dft_calc.results.get('stress', None)
dft_calc_std = dft_calc.results.get('stds', None)
print(f'dft_calc_e: {dft_calc_e}')
