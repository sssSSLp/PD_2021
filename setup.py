import codecs
from setuptools import find_packages
from setuptools import setup

install_requires = [
    'gitpython',
    'numpy',
    'pandas',
    'optuna==1.3.0',
    'scikit-learn==0.22.2',
    "PyYAML"
]

extras = {
    'test': [
        'flake8',
        'pytest',
    ]
}

setup(name='pdc',
      version='0.0.1',
      description='Parkinsons Disease Classifier',
      long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Shuji Suzuki',
      author_email='ssuzuki@preferred.jp',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require=extras)
