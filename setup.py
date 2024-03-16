from setuptools import find_packages,setup
from typing import List

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""

setup(
    name='ocr_captcha',
    version='0.0.1',
    author='Aadarsh Kumar Singh',
    author_email='aadarshkr.singh.cd.ece21@itbhu.ac.in',
    install_requires=[],
    packages=find_packages()
)