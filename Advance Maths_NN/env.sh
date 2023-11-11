#! /bin/bash
# latest packages as of 2023-04-06

python3 -m venv $(pwd)/venv/protein
source $(pwd)/venv/protein/bin/activate
pip install --upgrade pip
pip install pandas
pip install numpy
pip install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo 
echo 'Done! as of 2023-04-06 lastest required packages are installed succesfully.'
