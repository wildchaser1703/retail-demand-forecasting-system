from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="retail-demand-forecasting",
    version="1.0.0",
    author="Toshali Mohapatra",
    description="Industry-standard retail demand forecasting system with multiple ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wildchaser1703/retail-demand-forecasting-system",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "pre-commit>=3.3.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "forecast-train=scripts.run_pipeline:main",
            "forecast-api=api.main:start_server",
        ],
    },
)
