module load SciPy-bundle/2022.05-foss-2022a
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

source ../my_python/bin/activate

pip install torchvision
pip install --no-cache-dir --no-build-isolation flair