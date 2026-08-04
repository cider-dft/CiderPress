#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

import os
from importlib.resources import files

import joblib
import yaml

from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2

CIDER23X_MODELS = (
    "CIDER23X_SL_GGA",
    "CIDER23X_SL_MGGA",
    "CIDER23X_NL_GGA",
    "CIDER23X_NL_MGGA",
    "CIDER23X_NL_MGGA_PBE",
    "CIDER23X_NL_MGGA_DTR",
)

CIDER24X_MODELS = (
    "CIDER24Xne",
    "CIDER24Xe",
)

CIDER26XC_MODELS = (
    "CIDER26XCCHEM",
    "CIDER26XCCHEMD4",
    "CIDER26XCSURFSCI",
)

BUILTIN_MODELS = CIDER23X_MODELS + CIDER24X_MODELS + CIDER26XC_MODELS


def _get_builtin_model_name(value):
    if value in BUILTIN_MODELS:
        return value
    if value.endswith(".yaml") and value[:-5] in BUILTIN_MODELS:
        return value[:-5]
    return None


def load_cider_model(mlfunc, mlfunc_format=None):
    if isinstance(mlfunc, os.PathLike):
        mlfunc = os.fspath(mlfunc)
    if isinstance(mlfunc, str):
        resource = None
        if not os.path.exists(mlfunc):
            builtin_name = _get_builtin_model_name(mlfunc)
            if builtin_name is not None:
                resource = files("ciderpress.data.functionals").joinpath(
                    builtin_name + ".yaml"
                )
                if mlfunc_format not in (None, "yaml"):
                    raise ValueError("Built-in CIDER models use YAML format")
                mlfunc_format = "yaml"
        if mlfunc_format is None:
            if mlfunc.endswith(".yaml"):
                mlfunc_format = "yaml"
            elif mlfunc.endswith(".joblib"):
                mlfunc_format = "joblib"
            else:
                raise ValueError("Unsupported file format")
        if mlfunc_format == "yaml":
            context = resource.open("r") if resource is not None else open(mlfunc, "r")
            try:
                with context as f:
                    mlfunc = yaml.load(f, Loader=yaml.CLoader)
            except ModuleNotFoundError as exc:
                if exc.name == "torch":
                    raise ModuleNotFoundError(
                        "This CIDER model requires PyTorch. Install the optional "
                        "dependency with `pip install 'ciderpress[cider24]'`, or "
                        "install a platform-specific PyTorch build before loading it."
                    ) from exc
                raise
        elif mlfunc_format == "joblib":
            mlfunc = joblib.load(mlfunc)
        else:
            raise ValueError("Unsupported file format")
    if not isinstance(mlfunc, (MappedXC, MappedXC2)):
        raise ValueError("mlfunc must be MappedXC or MappedXC2")
    return mlfunc


def get_slxc_settings(xc, xkernel, ckernel, xmix):
    if xc is None:
        # xc is another way to specify non-mixed part of kernel
        xc = ""
    if ckernel is not None:
        xc = ckernel + " + " + xc
    if xkernel is not None:
        xc = "{} * {} + {}".format(1 - xmix, xkernel, xc)
    if xc.endswith(" + "):
        xc = xc[:-3]
    return xc
