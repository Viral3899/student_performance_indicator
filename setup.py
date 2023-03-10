from setuptools import find_packages,setup
from typing import List



PROJECT_NAME="student_project"
VERSION='0.0.1'
AUTHOR='Viral'
AUTHOR_EMAIL='viralsherathiya1008@gmail.com'
HYPHEN_E_DOT='-e .'


def get_requirements(file_path:str,)->List[str]:
    '''
    This Function Returns the List of Requirements
    
    '''

    requirements=[]
    with open ('requirements.txt') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]


        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    
    return requirements






setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')


)