import pathlib

import setuptools

setuptools.setup(

    name="cfd",
    version="0.1.0",
    description="Fluid Dynamics",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Chris DeGrendele"
)