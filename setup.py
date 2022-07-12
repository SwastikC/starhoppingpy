from setuptools import setup, find_packages

setup(
    name='starhoppingpy',
    version='1.0.0',
    url='https://github.com/SwastikC/starhoppingpy.git',
    author='Swastik C',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
