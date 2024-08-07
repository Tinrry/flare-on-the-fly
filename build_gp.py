import numpy as np
from flare.io import otf_parser
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp.sparse_gp import SGP_Wrapper
import warnings

try:
    from flare.bffs.sgp._C_flare import SparseGP, NormalizedDotProduct, B2, DotProduct
except ImportError as e:
    Warning.warn(f"Could not import _C_flare: {e.__class__.__name__}: {e}.")

model_dir = './'


def load_sgp():
    # load lammps-otf_flare.json
    sgp_file = f'{model_dir}/offline_flare.json'

    import json
    # set up ASE flare calculator, it will add environment variable to the system, but it will cause many times waste
    with open(sgp_file, 'r') as f:
        gp_dct = json.loads(f.readline())
        if gp_dct.get("class", None) == "SGP_Calculator":
            flare_calc, kernels = SGP_Calculator.from_file(sgp_file)
        else:
            sgp, kernels = SGP_Wrapper.from_file(sgp_file)
            flare_calc = SGP_Calculator(sgp)
    return flare_calc


def read_xyz(atoms_config, atoms_index = ':'):
    from ase.io import read
    atoms_file = atoms_config.get('file')
    atoms_format = atoms_config.get('format', None)
    if atoms_format is not None:
        structure_set = read(atoms_file, format=atoms_format, index=atoms_index)
    else:
        structure_set = read(atoms_file, index=atoms_index)
    return structure_set


def load_atoms():
    from flare.scripts.otf_train import get_super_cell, get_dft_calc
    import yaml

    # config the dft calculator and supercell
    config_file = f'{model_dir}/offline.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        otf_config = config["otf"]
    # this read the dataset symbol and positions
    # 这里只读所有structure中的最后一个
    # super_cells = read_xyz(config["supercell"])
    super_cell = get_super_cell(config["supercell"])
    super_cells = [super_cell]
    dft_calc = get_dft_calc(config["dft_calc"])
    return super_cells, dft_calc, otf_config


def predict(flare_calc, super_cell, dft_calc, otf_config):
    # predict the energy and force fro structures, 使用方法与ASE相同
    # if we use flare calculator, we must wrap it in OTF, use FLARE_Atoms
    from flare.scripts.otf_train import OTF

    # model set up
    otf = OTF(atoms=super_cell, flare_calc=flare_calc, dft_calc=dft_calc, **otf_config)
    # write the energy and forces to the atoms
    # otf.atoms.set_calculator(flare_calc)

    
    # predict the energy and force
    # 调用get_potential_energy()或get_forces()会触发一个自洽计算，并输出大量的输出文本。

    # otf.atoms.get_potential_energy()
    # otf.atoms.get_forces()
    # otf.atoms.get_stress()
    # otf.atoms.write('sgp_flare.xyz', format='extxyz')
    # print(f'otf.atoms.get_stress: {otf.atoms.get_stress()}')
    flare_calc_e = flare_calc.results.get('energy', None)
    flare_calc_f = flare_calc.results.get('forces', None)
    flare_calc_s = flare_calc.results.get('stress', None)
    # print(f'flare_calc_s: {flare_calc_s}')      


    # 在OTF中 atoms 有一个属性，flare results，
    # 这个属性是一个字典，里面存储了energy和forces
    # flare_atoms_e = otf.atoms.energy
    print(f'flare_calc_e: {flare_calc_e}')      # -332.18692436639685
    print(f'potential_energy: {otf.atoms.get_potential_energy()}')     
    # print(f'flare_calc_f: {flare_calc_f}')      # [[ -3.07878154e-01 -1.00920507e-01  2.26388432e-01] ...]
    print(f'flare_calc_s: {flare_calc_s}')      # [-1.16636430e-03 -1.22592367e-03 -3.44829727e-03 -9.08749919e-06 -1.05114523e-04 -3.70648447e-05]
    print(f'flare_calc_f shape: {np.array(flare_calc_f).shape}')  
    -325.2556407470256


if __name__ == '__main__':
    flare_calc = load_sgp()
    super_cells, dft_calc, otf_config = load_atoms()
    for atoms in super_cells:
        # predict(flare_calc, atoms, dft_calc, otf_config)
        print(f'atoms: {atoms}')
        break