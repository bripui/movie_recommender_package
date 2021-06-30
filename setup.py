from setuptools import setup, find_packages
import os

def open_file(fname):
    """helper function to open a local file"""
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(
    name='movierecommenderpackage',
    version='0.0.1',
    author='Britta Puyn',
    author_email='britta.puyn@outlook.de',
    packages=find_packages(),
    url='https://github.com/bripui/movie_recommender_package',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    package_data= {
        'movierecommenderpackage': ['data/ml-latest-small/*.csv', 'models/*.sav']
    },
    description='Implementation of various collaborative filtering methods',
    long_description=open_file('README.md').read(),
    # end-user dependencies for your library
    install_requires=[
        'pandas',
        'scikit-learn',
        'fuzzywuzzy',
        'python-Levenshtein'        
    ],
)
