from setuptools import find_packages, setup


setup(
    author="MSR-India Systems Group; Systems for AI Lab, Georgia Tech",
    python_requires='>=3.10',
    description="Odyn PD-disaggregation simulator",
    include_package_data=True,
    keywords='odyn-simulator',
    name='odyn-simulator',
    packages=find_packages(include=['vidur', 'vidur.*']),
    version='0.0.1',
)
