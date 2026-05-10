
from setuptools import setup, find_packages

setup(
    name='jetito',
    version='0.5.0',
    description='Classes and methods for data analyis of experimental results at JETi-200 laser system.',
    author='Felipe C. Salgado',
    author_email='felipe.salgado@uni-jena.de',
    url='https://github.com/felipecsalgado/jetito',
    license="CC-BY-NC-SA",
    packages=find_packages(include=['jetito', 'jetito.*', 'jetito.ebeam.*', 'jetito.ebeam']),
    install_requires=[
        'numpy>=1.23.0',
        'matplotlib>=3.6.0',
        'scipy>=1.9.0',
        'opencv-python>=4.6.0',
        'pypng>=0.20220715.0',
        'imageio>=2.22.0',
        'Pillow>=9.3.0',
        'PeakUtils>=1.3.4',
        'scikit-image>=0.19.3',
        'pandas>=1.5.0',
        'natsort>=8.2.0',
        'pymc>=5.0.0',
        'arviz>=0.13.0',
        'tqdm>=4.64.0'
    ],
    keywords = [
        'jeti',
        'lwfa',
        'electron'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    python_requires=">=3.10",
)

