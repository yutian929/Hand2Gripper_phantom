cd ARX_R5_python/
git clone https://github.com/pybind/pybind11.git && cd pybind11 && mkdir build && cd build && cmake .. && make && sudo make install
cd ../../
./build.sh
echo LD_LIBRARY_PATH=$(pwd)/bimanual/api/arx_r5_src:$LD_LIBRARY_PATH >> ~/.bashrc
echo LD_LIBRARY_PATH=$(pwd)/bimanual/api:$LD_LIBRARY_PATH >> ~/.bashrc
echo LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH >> ~/.bashrc
pip install -e .
cd ..