# this initialize the flare environment, for use with conda
# ------ we must use python -c "import flare.bffs.sgp._C_flare" after the openblas, go through the import line, then vasp -------
module load conda
conda activate flare

module load openblas
python -c "import flare.bffs.sgp._C_flare" 

module load vasp
export VASP_PP_PATH=/data/projects/vasp/
python -c "import flare.bffs.sgp._C_flare" 
