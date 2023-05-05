from setuptools import find_packages, setup

VERSION = {}  # type: ignore
with open("ttt/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

ml = [
    "transformers==3.1.0",  # PTMs
    "torchvision",  # For CV datasets and transformers
    "sklearn",  # For statistical learning
    "gensim",  # For word embedding, dictionary, LDAs
    "tensorboard",  # For visualization
    "sklearn_crfsuite",  # for sklearn-like CRF
]

tools = [
    "pyhocon",  # For config file
    "registrable",
    "gputil",  # Inspect gpu info
    "pydantic",  # For base model
    "pandas",  #
    "jupyterlab",  # Notebook
    "matplotlib",  # Plotting
    "wordcloud",  # Cluster visualization
    "tqdm",  # progress bars
    "requests",  # Making http request (for testing and downloading)
    "jieba",  # Chinese word segmentation
    "zhon",  #
    "xlrd",  # For reading excel
    "typer[all]",  # CLIs
    "crc32c",
    #"pyahocorasick @ git+https://github.com/WojciechMula/pyahocorasick#egg=pyahocorasick",
    #"fastText @ git+https://github.com/facebookresearch/fastText.git@b64e359d5485dda4b4b5074494155d18e25c8d13#egg=fastText",
]

cv = [
    "Pillow",  # For image preprocessing
]

serving = [
    "fastapi",  # For Rest apis
    "aiofiles",  # For async file IO
    "jinja2",  # Html template
    "python-multipart",  # File uploading
    "uvicorn",  # Web container
]

testing = [
    "pytest",  # For running tests
    "pytest-benchmark",  # For running benchmarks
]

quality = [
    "mypy",  # Static type checking
    "black",  # Automatic code formatting
    "flake8",  # Checks style, syntax, and other useful errors
    "pytest-cov",  # Allows generation of coverage reports with pytest
    "coverage",  # Allows codecov to generate coverage reports
    "codecov",  # Allows codecov to generate coverage reports
]

install_requires = ml + tools + cv + serving + testing + quality

setup(
    name="ttt",
    version=VERSION["VERSION"],
    description="the true package call man",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "benchmarks", "benchmarks.*"]),
    author="siyan",
    author_email="xsy233@gmail.com",
    install_requires=install_requires,
    entry_points={"console_scripts": ["ttt-cli=ttt.__main__:main"]},
    include_package_data=True,
    python_requires=">=3.6.1",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License"
    ]
)
