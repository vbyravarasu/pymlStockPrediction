from setuptools import setup, find_packages

install_requires = [
    'pandas',
    'numpy',
    'scikit-learn',
    'statsmodels',
    'matplotlib',
]

setup(
    name='pymlStockPrediction',
    version='0.0.1',
    author='Yuting Wen',
    author_email='yutingyw@gmail.com',
    url='https://github.com/yutingyw/pymlStockPrediction',
    license='GPL',
    packages=find_packages(),
    install_requires=install_requires,
)