# pylint: disable=all

from setuptools import find_packages, setup

from owlite_core.constants import OWLITE_GIT_REPO_URL, OWLITE_VERSION


def requirements() -> list[str]:
    return [
        "torch>=2.0,<2.2",
        "onnxruntime",
        "onnxsim",
        "onnx_graphsurgeon",
        "colored",
        "yacs",
        "tabulate",
        "requests",
        "tqdm",
        "pydantic",
    ]


setup(
    name="owlite",
    version=OWLITE_VERSION,
    description="OwLite - No-Code AI compression Toolkit",
    url=OWLITE_GIT_REPO_URL,
    author="SqueezeBits Inc.",
    author_email="owlite@squeezebits.com",
    install_requires=requirements(),
    packages=find_packages(exclude=("test", "scripts")),
    python_requires="~=3.10.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["torch", "onnx", "graph", "quantization"],
    entry_points={
        "console_scripts": ["owlite=owlite_core.cli.owlite_cli:main"],
    },
)
