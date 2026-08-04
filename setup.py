import os
import sys
from glob import glob

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

# TODO not using wheel yet, but plan to do so eventually
from wheel.bdist_wheel import bdist_wheel


def get_version():
    topdir = os.path.abspath(os.path.join(__file__, ".."))
    with open(os.path.join(topdir, "ciderpress", "__init__.py"), "r") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise ValueError("Version string not found")


VERSION = get_version()


def get_platform():
    from sysconfig import get_platform

    platform = get_platform()
    # TODO might want to add darwin OSX support like PySCF
    # but only after officially adding OSX support
    return platform


class CMakeBuildPy(build_py):
    def _copy_built_libraries(self, src_dir):
        """Copy CMake outputs into the wheel's package tree.

        Shared libraries are intentionally excluded from the source
        distribution so a local build cannot contaminate a release archive.
        CMake writes the libraries into the extracted source tree, so they must
        be copied explicitly after the normal Python package files.
        """
        destinations = (
            (src_dir, os.path.join("ciderpress", "lib")),
            (
                os.path.join(src_dir, "deps", "lib"),
                os.path.join("ciderpress", "lib", "deps", "lib"),
            ),
            (
                os.path.join(src_dir, "deps", "lib64"),
                os.path.join("ciderpress", "lib", "deps", "lib64"),
            ),
        )
        for source_dir, relative_destination in destinations:
            libraries = set(glob(os.path.join(source_dir, "*.so*")))
            libraries.update(glob(os.path.join(source_dir, "*.dylib*")))
            if not libraries:
                continue
            destination = os.path.join(self.build_lib, relative_destination)
            self.mkpath(destination)
            for library in sorted(libraries):
                if os.path.isfile(library):
                    self.copy_file(library, destination)

    def run(self):
        self.plat_name = get_platform()
        self.build_base = "build"
        self.build_lib = os.path.join(self.build_base, "lib")
        self.build_temp = os.path.join(self.build_base, f"temp.{self.plat_name}")

        self.announce("Configuring extensions", level=3)
        src_dir = os.path.abspath(os.path.join(__file__, "..", "ciderpress", "lib"))
        cmd = [
            "cmake",
            f"-S{src_dir}",
            f"-B{self.build_temp}",
            "-DCMAKE_PREFIX_PATH={}".format(sys.base_prefix),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        configure_args = os.getenv("CMAKE_CONFIGURE_ARGS")
        if configure_args:
            cmd.extend(configure_args.split(" "))
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ["cmake", "--build", self.build_temp, "-j2"]
        build_args = os.getenv("CMAKE_BUILD_ARGS")
        if build_args:
            cmd.extend(build_args.split(" "))
        if self.dry_run:
            self.announce(" ".join(cmd))
        else:
            self.spawn(cmd)
        self.editable_mode = False
        super().run()
        self._copy_built_libraries(src_dir)


# NOTE note trying to support wheal yet, but including
# these code block from PySCF setup.py for future.
initialize_options_1 = bdist_wheel.initialize_options


def initialize_with_default_plat_name(self):
    initialize_options_1(self)
    self.plat_name = get_platform()
    self.plat_name_supplied = True


bdist_wheel.initialize_options = initialize_with_default_plat_name

# from PySCF setup.py
try:
    from setuptools.command.bdist_wheel import bdist_wheel

    initialize_options_2 = bdist_wheel.initialize_options

    def initialize_with_default_plat_name(self):
        initialize_options_2(self)
        self.plat_name = get_platform()
        self.plat_name_supplied = True

    bdist_wheel.initialize_options = initialize_with_default_plat_name
except ImportError:
    pass

setup(
    version=VERSION,
    include_package_data=True,
    packages=find_packages(exclude=["*test*", "*examples*"]),
    cmdclass={"build_py": CMakeBuildPy},
)
