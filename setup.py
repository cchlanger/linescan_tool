import setuptools

# read the contents of the README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="linescan_tool",
    version="0.1",
    author="Christoph Langer",
    author_email="cchlanger@gmail.com",
    description="A linescan image analysis tool",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cchlanger/linescan_tool",
    packages=setuptools.find_packages(),
    install_requires=[
        "read-roi",
        "seaborn",
        "pandas",
        "skimage",
        "scipy",
        "matplotlib",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)