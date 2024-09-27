import setuptools
from setuptools.command.build_ext import build_ext


class build_ext_(build_ext):
    def run(self):
        pass


setuptools.setup(
    ext_modules=[
        # just a dummy extension for platform specific build
        setuptools.Extension(
            name="owlite.capi",
            sources=[],
        )
    ],
    cmdclass={"build_ext": build_ext_},
)
