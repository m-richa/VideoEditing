USER="$(whoami)"

PYTHON_ENV=/home/${USER}/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source ${PYTHON_ENV}

conda activate champ
cd champ/

echo "Generate inpainted image"
python inpainting.py

echo "Generate video"
python inference.py 