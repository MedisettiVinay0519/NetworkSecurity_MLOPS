from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    requirements_lst:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            # Read the lines from the file and process them
            lines= file.readlines()
           # Iterate through each line and add it to the requirements list
            for line in lines:
                requirements=line.strip()
                # Check if the line is not empty and does not start with '-e .'
                if requirements and  requirements!= '-e .':
                    requirements_lst.append(requirements)
    except FileNotFoundError:
        print("requirements.txt file not found.")

    return requirements_lst        
 
setup(
    name='Network_Secuirty',
    version='0.0.1',
    author='Vinay Medisetti',
    author_email="vinaymedisetti05@gamil.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

