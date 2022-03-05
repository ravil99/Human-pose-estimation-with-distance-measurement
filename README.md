# Installation

```
python -m venv env
. env/bin/activate
pip install -r requirements.txt
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone https://github.com/isarandi/poseviz.git
cd poseviz
pip install .
cd ../
rm -rf poseviz/
```