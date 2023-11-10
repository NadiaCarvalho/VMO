from setuptools import setup, find_packages, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

try:
    requirements = open('requirements.txt').read().splitlines()
except:
    requirements = []
    print("Couldn't install requirements")

    
setup(
    name="syvmo",
    version='0.0.1',
    description='SyVMO - ',
    long_description=open('README.md').read(),
    author="Nádia Carvalho",
    author_email="nadiacarvalho118@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    keywords='vmo, syvmo',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
