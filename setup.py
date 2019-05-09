from setuptools import setup
with open('requirements.txt','r') as f:
    requirements = f.readlines()

setup(
   name='seqscout',
   version='1.0',
   description='Package for discovering interesting sequences',
   author='Blind',
   packages=['seqscout'],
   install_requires=requirements,
)
