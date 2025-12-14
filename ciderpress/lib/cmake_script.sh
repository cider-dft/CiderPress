export MY_PYTHON_LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export MY_PYTHON_INCDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')

echo $MY_PYTHON_LIBDIR $MY_PYTHON_INCDIR

export LIBRARY_PATH=$LIBRARY_PATH:$CONDA_PREFIX/lib

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_WITH_MKL=ON \
      -DBUILD_LIBXC=OFF \
      -DBUILD_WITH_MPI=ON \
      -DPython_EXECUTABLE=$(which python) \
      ..
make
