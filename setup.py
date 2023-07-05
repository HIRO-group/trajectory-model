import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setuptools.setup(
    name='trajectory_model',
    author="Ava Abderezaei",
    author_email='ava.abderezaei@colorado.edu',
    version='1.0.0',
    description='Trajectory Model for slosh free path planning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HIRO-group/trajectory-model',
    packages=setuptools.find_packages(where='trajectory_model*'),
    install_requires=['numpy==1.24.2', 'tensorflow==2.13.0-rc1', 'urllib3==1.26.13'],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
     python_requires='>=3.8'
)