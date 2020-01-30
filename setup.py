import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicon",
    version="1.0.6",
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
        'pandas',
        'numpy',
        'networkx',
        'matplotlib',
        'scipy',
        'gseapy',
        'seaborn',
        'mygene',
        'scikit_learn'
    ],

)
