import os
import subprocess
import sys

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

star_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "STAR")
if not os.path.isdir(star_dir):
    subprocess.check_call(["git", "clone", "https://github.com/NJU-PCALab/STAR.git", star_dir])
