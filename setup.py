from setuptools import setup
from os import path
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='decision-tree-classifier',
    version='1.0.0',
    description='Decision-tree classifier test',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='ssh://git@git.pega.io:7999/ds/oppfinder_poc.git',
    author='Natallia Kokash',
    author_email='natallia.kokash@pega.com',
    keywords='machine learning classifier decision tree',
    py_modules=['main', 'iris', 'utils', 'preprocess', 'c45'],
    python_requires='>=3.0.*',
    install_requires=['scikit-learn', 'numpy', 'pandas', 'matplotlib'],
    entry_points={
        'console_scripts': [
            'test_mobile=main:test_mobile',
            'test_iris=iris:test_iris',
            'test_iris_c45=iris:test_iris_c45'
        ]}
)