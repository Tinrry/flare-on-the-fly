from ase.build import molecule
from ase.collections import g2
from ase.build.molecule import extra

from ase.spacegroup import crystal


print(g2.names)
print(extra.keys())
water = molecule('H2O')
print(f'Water {water}')

bery1 = crystal('Be', [(0, 0, 0)], spacegroup=152, cellpar=[2.29, 2.29, 3.58, 90, 90, 120])