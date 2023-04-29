conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

conda create -n hq python=3.8
conda activate hq
pip3 install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt --use-pep517

mkdir -p data/gt
mkdir data/lq
mkdir data/hq
mkdir -p face_detection/detection/sfd
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "face_detection/detection/sfd/s3fd.pth"

python inference.py --static True --checkpoint_path "checkpoints/wav2lip.pth" --face videos/BIBA.mp4.png --audio videos/BIBA_30s.mp4 --outfile output/1.mp4

python inference1.py --checkpoint_path "checkpoints/wav2lip.pth" --face videos/BIBA_30s.mp4 --audio videos/BIBA_30s.mp4 --outpath output

