module purge
module use /apps/leuven/skylake/2019b/modules/all
module load cuDNN/8.1.1.33-CUDA-11.2.2
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
conda activate PythonGPU