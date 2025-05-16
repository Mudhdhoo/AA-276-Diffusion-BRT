from setuptools import setup, find_packages

setup(
    name="diffusion-brt",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "wandb",
        "pandas"
    ]
) 