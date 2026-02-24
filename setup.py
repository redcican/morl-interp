from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="morl-interpretable",
    version="1.0.0",
    author="Shikun Chen, Yangguang Liu",
    description=(
        "Multi-Objective Evolutionary Reinforcement Learning "
        "for Pareto-Optimal Interpretable Policies"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous-morl-interp/morl-interp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "morl-train=experiments.train:main",
            "morl-evaluate=experiments.evaluate:main",
        ],
    },
)
