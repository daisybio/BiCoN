import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicon",
    version="1.2.14",
    author="Olga Lazareva",
    author_email="olga.lazareva@tum.de",
    description="BiCoN - a package for network-constrained biclustering of omics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biomedbigdata/BiCoN",
    packages=setuptools.find_packages(exclude=['test.py']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.0.1',
        'numpy>=1.18.1',
        'networkx>=2.3',
        'matplotlib>=3.1.0',
        'scipy>=1.3.0',
        'gseapy>=0.9.15',
        'seaborn>=0.9.0',
        'mygene>=3.1.0',
        'scikit_learn>=0.22'
    ],

)
