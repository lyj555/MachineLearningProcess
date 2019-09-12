# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='model-helper',
    version='0.0.2',
    url='http://igit.58corp.com/AIgroups/ModelHelper',
    packages=find_packages(),
    author='liuyongjie05',
    author_email="liuyongjie05@58.com",
    description='Model Helper',
    long_description='A Python implementation of model helper, with feature selection, hyper-parameter tune and feature engineering',
    # download_url='https://github.com/fmfn/BayesianOptimization/tarball/0.6',
    install_requires=[
        "pandas >= 0.23.0",
        "numpy >= 1.10.0",
        "scikit-learn >= 0.19.0",
        "shap >= 0.30.0"
    ],
)
