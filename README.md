# icp
[WAFR24] Interaction-aware Conformal Prediction for Crowd Navigation

# Setup
Run
```
conda create -n icp python=3.9.0
conda activate icp
pip install numpy
pip install pandas
pip install torch==2.0.1
pip install gym
pip install tensorflow
pip install matplotlib
pip install imageio[ffmpeg]
pip install casadi
```

Then run
```
git clone https://github.com/openai/baselines.git
cd baselines/
pip install -e .
cd ..
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2/
pip install Cython
python setup.py build
python setup.py install
cd ..
```

Note 1: If you encounter the error similar to below when running `python setup.py build`
```
CMake Error at CMakeLists.txt:33 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```
Remove `Python-RVO2/` and collect the repo again.
```
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2/
```
Adjust the line in `icp/Python-RVO2/CMakeLists.txt`
```
cmake_minimum_required(VERSION 2.8)
```
to
```
cmake_minimum_required(VERSION 3.5)
```
and then run
```
pip install Cython
python setup.py build
python setup.py install
cd ..
```
Check if installed successfully by running `import rvo2` in python in the `icp` environment.
Note 2: Replace `np.bool` with `bool` in `icp/baselines`.


# How to run
```
conda activate icp
python wafr_evaluate_algorithms.py -a icp -i 3 -n 10 -p 5 -r -s 8 --calibration_size 8 --calibration_envs_num_processes 8
```

Note 3: If you encounter error related to `RuntimeError: Numpy is not available`, run
```
conda activate icp
pip install "numpy<2"
python wafr_evaluate_algorithms.py -a icp -i 3 -n 10 -p 5 -r -s 8 --calibration_size 8 --calibration_envs_num_processes 8
```



