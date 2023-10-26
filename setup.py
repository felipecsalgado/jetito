
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
    #install_requires=[
        #'numpy>=1.18.5',
        #'matplotlib>=3.0.3',
        #'cv2',
        #'png',
        #'imageio',
        #'scipy'
    #],
    keywords = [
        'jeti',
        'lwfa',
        'electron'
    ],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    python_requires=">=3.5",
)

