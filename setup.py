from setuptools import setup
from setuptools import find_packages

setup(name='block_diagonal_gcn',
      version='0.1',
      description='Block Diagonal Graph Convolutional Network in PyTorch',
      author='Po-wei Harn, Sai Deepthi',
      author_email='harnpowei@gmail.com',
      url='',
      download_url='',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy',
                        'matplotlib',
                        'sklearn',
                        'mlxtend'
                        ],
      package_data={'block_diagonal_gcn': ['README.md']},
      packages=find_packages())