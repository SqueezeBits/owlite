# pylint: disable=missing-function-docstring, missing-module-docstring
import platform

from setuptools import find_packages, setup


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_ppc64le() -> bool:
    return platform.machine() == "ppc64le"


def is_cygwin() -> bool:
    return platform.system().startswith("CYGWIN_NT")


def requirements() -> list[str]:
    return [
        "torch>=2",
        "onnxruntime",
        "onnxsim",
        "onnx_graphsurgeon",
        "colored",
        "yacs",
        "tabulate",
        "requests",
        "tqdm",
    ]


setup(
    name="owlite",
    version="v0.1",
    description="OwLite - No-Code AI compression Toolkit",
    url="https://bitbucket.org/squeezebits/owlitetorch",
    author="SqueezeBits Inc.",
    author_email="owlite@squeezebits.com",
    install_requires=requirements(),
    packages=find_packages(exclude=("test",)),
    python_requires="~=3.10.0",
    classifiers=[
        "Development Status :: 1 - Planning",
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
