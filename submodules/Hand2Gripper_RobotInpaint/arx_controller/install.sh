cd ARX_R5_python/
# 如果存在pybind11目录了，就删掉build，重新build

if [ ! -d "pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git
fi

cd pybind11
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build && cmake .. && make && sudo make install
cd ../../  # ARX_R5_python/
cd bimanual
if [ -d "build" ]; then
    rm -rf build
fi
./build.sh

# 如果LD_LIBRARY_PATH路径已经有了，就不要重复写
if ! grep -qF "ARX_R5_python bimanual package" ~/.bashrc; then
    echo "# ARX_R5_python bimanual package" >> ~/.bashrc
fi

# 注意 \$LD_LIBRARY_PATH
CUR_DIR=$(pwd)
if ! grep -qF "$CUR_DIR/bimanual/api/arx_r5_src:" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=$CUR_DIR/bimanual/api/arx_r5_src:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi
if ! grep -qF "$CUR_DIR/bimanual/api:" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=$CUR_DIR/bimanual/api:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi
if ! grep -qF "/usr/local/lib:" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi

cd ../ # ARX_R5_python/
pip install -e .
cd ..