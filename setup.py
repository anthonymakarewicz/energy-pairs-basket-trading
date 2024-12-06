from setuptools import setup, find_packages

setup(
    name="Pairs trading",          
    version="0.1",
    description="A simpel pairs trading project",
    author="Anthony Makarewicz",
    author_email="anthonymakarewicz@gmail.com",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.6",
)