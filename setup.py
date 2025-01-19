from setuptools import setup, find_packages

setup(
    name="swarmsim",  # Replace with your package name
    version="0.1.0",
    author="Stefano Covone, Italo Napolitano, Davide Salzano",
    author_email="s.covone@ssmeridionale.it",
    description="Large-scale multi-agent simulator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SINCROgroup/large_scale_multi_agent_simulator",
    packages=find_packages(),  # Automatically find packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").read().splitlines(),
)
