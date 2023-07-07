
from setuptools import setup, find_packages

setup(
    name='jetito',
    version='0.3.0',
    description='Classes and methods for data analyis of experimental results at JETi-200 laser system.',
    author='Felipe C. Salgado',
    author_email='felipe.salgado@uni-jena.de',
    url='https://github.com/hixps/jetito',
    license="CC-BY-NC-SA",
    packages=find_packages(include=['jetito', 'jetito.*']),
    install_requires=[
        'numpy>=1.18.5',
        'matplotlib>=3.0.3',
        'cv2>=4.4.0',
        'png>=0.0.20',
        'imageio>=2.9.0',
        'scipy>=1.4.1'
    ],
    keywords = [
        'jeti',
        'lwfa',
        'electron'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

