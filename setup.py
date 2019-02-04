import os
from distutils.core import setup
from pathlib import Path


def read_requirements():
    "Read requirements file and returns those lines corresponding to library dependencies"
    pwd = Path(os.path.dirname(os.path.abspath(__file__)))
    lines = []
    with open(pwd / 'requirements.txt', 'rt') as fh:
        for line in fh:
            line = line.strip()
            pos = line.find('#')
            if pos >= 0:
                line = line[:pos].strip()
            if not line:
                continue
            lines.append(line)
    return lines


requirements = read_requirements()

setup(
    name='pywde',
    version='0.1',
    packages=['pywde'],
    url='',
    license='',
    author='Carlos Aya',
    author_email='',
    description='Wavelet density estimation in Python',
    py_modules=['pywde'],
    install_requires=requirements,
    setup_requires=[]
)
