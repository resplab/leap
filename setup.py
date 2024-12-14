from setuptools import setup

setup(
    name="leap",
    version="1.0",
    description="Lifetime Exposures and Asthma outcomes Projection (LEAP)",
    author="Tae Yoon (Harry) Lee, Ainsleigh Hill",
    author_email="ainsleighhill@gmail.com",
    packages=["leap"],
    install_requires=[
      line.strip() for line in open("requirements.txt")
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
        "docs": [line.strip() for line in open("requirements-docs.txt")]
    },
    include_package_data=True,
    package_data={
        "": ["tests/data/", "processed_data/"]
    },
    entry_points={
        "console_scripts": ["leap=leap.main:run_main"]
    },
)
