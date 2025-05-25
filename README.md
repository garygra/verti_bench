<h1 align="center">
Verti-Bench: A General and Scalable Off-Road Mobility Benchmark for Vertically Challenging Terrain 
</h1>

<div align="center">

Robotics: Science and Systems (RSS) 2025

[[Website]](https://cs.gmu.edu/~xiao/Research/Verti-Bench/)
[[Arxiv]](https://arxiv.org/pdf/2502.11426)
[[Video]](https://www.youtube.com/watch?v=O9VlMg3tnvo)

<p align="center">
    <img src="assets/gmu.png" height=50"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="assets/RobotiXX_gmu.gif" height=50">
</p>


[![Chrono](https://img.shields.io/badge/Chrono-9.0.1-brightgreen.svg)](https://projectchrono.org/) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

<p align="center">
    <img src="assets/Verti_bench.png">
</p>

</div>

### TODO
- [x] Release code pipeline
- [x] Release ten off-road mobility systems
- [ ] Release different scale vehicles
- [ ] Release datasets from expert demonstration, random exploration, failure cases

# Installation
There are two options for installing PyChrono on your computer. The first one uses a prebuilt conda packages and is the recommended way. The second one is for users who need to build the full library from the C++ source.

### Dependencies:
> Ubuntu 22.04 or above.

> ROS Humble.

> Miniconda for Linux.

> NVIDIA CUDA driver & Toolkit 11.8 (use `nvcc --version` & `nvidia-smi` to test installation).

### A) Pychrono 9.0.1 from Conda 
1. Add the `conda-forge` channel to the list of channels:
```
conda config --add channels https://conda.anaconda.org/conda-forge
```
2. Create a `chrono9` environment with Python 3.9:
```
conda create -n chrono9 python=3.9
```
Then activate that environment:
```
conda activate chrono9
```
so that all subsequent conda commands occur within that environment.

3. Install the necessary dependencies:

**Attention**: Install the following packages using the versions specified below, in the order given, and before installing the `PyChrono` conda package itself, which is done in a subsequent step!

- Installing gymnasium
```
pip install gymnasium
```

- Installing stable-baselines3
```
pip install stable-baselines3[extra] 
```
> [!NOTE]
> `stable-baselines3` installs nupmy as a dependency, so it is recomended to remove this installation and install your own version of numpy. Additionally, `pychrono` requires `numpy=1.24.0`, and it must be installed with conda as below.

- Intel MKL package (required for PyChrono demos using the Pardiso direct sparse linear solver, for Numpy, and for PythonOCC)
```
conda install -c conda-forge mkl=2020
```

- Numpy related package (required for the `Chrono::Sensor` module)
```
pip uninstall numpy
conda install -c conda-forge numpy=1.24.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml scipy
```

- Irrlicht, for run-time visualization
```
conda install -c conda-forge irrlicht=1.8.5
```

- Pythonocc-core, for `Chrono::Cascade` support
```
conda install -c conda-forge pythonocc-core=7.4.1
```

- Gnuplot, for graphing data
```
conda install conda-forge::gnuplot
```

- For `Chrono::Sensor` support, install GLFW
```
conda install -c conda-forge glfw
```

4. Install the `PyChrono 9.0.1` conda package from [here](https://drive.google.com/file/d/1bEiBawKRFip1th70EcP5w6Hr-nh6rXMP/view?usp=sharing) in the google drive:
```
conda install pychrono-9.0.1-py39_1.tar.bz2
```

5. After clone this codebase, add Verti-Bench data directory to path:
```
echo 'export CHRONO_DATA_DIR=$HOME/Documents/verti_bench/envs/data/' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:$HOME/Documents/' >> ~/.bashrc
```

6. Install ros2 humble related pkgs for systems
```
sudo apt install ros-humble-grid-map-msgs ros-humble-geometry-msgs
```

### B) Pychrono 9.0.1 from C++ API
1. Clone the 901 branch from [here](https://github.com/madhan001/chrono/tree/901) and update the submodule:
```
git clone -b 901 https://github.com/madhan001/chrono.git
cd chrono
git submodule update --init --recursive
```
2. Install the build dependencies:
- Install `cmake-gui`
```
sudo snap install cmake --classic
```
- Install `swig`: make sure `swig` version is above 4.0 and use `swig -version` to test installation
```
sudo apt update
sudo apt install swig
```
- Install necessary `gcc` tools
```
sudo apt update
sudo apt install gcc
sudo apt install build-essential
```
3. Add the `conda-forge` channel to the list of channels:
```
conda config --add channels https://conda.anaconda.org/conda-forge
```
4. Create a `chrono9` environment with Python 3.9:
```
conda create -n chrono9 python=3.9
```
Then activate that environment:
```
conda activate chrono9
```
so that all subsequent conda commands occur within that environment.

5. Install the necessary dependencies:

**Attention**: Install the following packages using the versions specified below, in the order given, and before installing the `PyChrono` conda package itself, which is done in a subsequent step!

- Installing gymnasium
```
pip install gymnasium
```

- Installing stable-baselines3
```
pip install stable-baselines3[extra] 
```
> [!NOTE]
> `stable-baselines3` installs nupmy as a dependency, so it is recomended to remove this installation and install your own version of numpy. Additionally, `pychrono` requires `numpy=1.24.0`, and it must be installed with conda as below.

- Numpy related package (required for the `Chrono::Sensor` module)
```
pip uninstall numpy
conda install -c conda-forge numpy=1.24.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml scipy
```
- **Deactivate conda virtual env**, install MPI (Message Passing Interface) for `Chrono::Synchrono` and `Chrono::Vehicle` modules
```
sudo apt-get update
sudo apt-get install libopenmpi-dev openmpi-bin
```
- Create libraries folder in chrono folder for third-party dependencies
```
cd chrono
mkdir libraries
```
- Install Eigen, GL and URDF: 

We strongly recommend using the latest Eigen3 version 3.4.0 by `buildEigen.sh` file under the `chrono/contrib/build-scripts/linux` folder.
```
cd contrib/build-scripts/linux
chmod +x buildEigen.sh buildGL.sh buildURDF.sh
./buildEigen.sh
./buildGL.sh
./buildURDF.sh
```
6. Install chrono modules from C++ source code:
- `Chrono::core` module: Get into your home code directory again to create build folder 
```
mkdir chrono_build
cd chrono_build
cmake-gui
```
After getting into cmake gui, use your own directory to set; Change `<YOUR_HOME_DIR>` to your real path!
```
#Where is the source code:
<YOUR_HOME_DIR>/Documents/chrono
#Preset
custom
#Where to build the binaries:
<YOUR_HOME_DIR>/Documents/chrono_build
```
For cmake gui, choose Unix Makefiles generator and add entries as below; Choose `cmake_build_type` as "Release".
```
#EIGEN3_INCLUDE_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/eigen/include/eigen3
```             
After clicking the configure button without error, close cmake gui. 
- `Chrono::sensor` module: Now we need to set CUDA architecture for this module. Directly from terminal, use below to enable CUDA
```
cmake -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="6.0" -DCUDA_ARCH_PTX="60" ../chrono
```
Download [OptiX](https://developer.nvidia.com/designworks/optix/downloads/legacy) - version 7.7 only in folder `<YOUR_HOME_DIR>/Documents/chrono/libraries`;

Install required development libraries for OpenGL and X11:
```
sudo apt update
sudo apt install libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```
Other related OpenGL compile dependency
```
sudo apt-get install libglu1-mesa-dev
sudo apt-get install freeglut3-dev
```
Use `cmake-gui` again to open cmake, click `ENABLE_MODULE_SENSOR` and configure button, then set entries as below and **unclick** the `use_cuda_nvrtc` option in the cmake gui
```
#GLEW_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/gl/lib/cmake/glew
#glfw3_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/gl/lib/cmake/glfw3
```
- `Chrono::irrlicht` module:
```
sudo apt install libirrlicht-dev
```
After installation, click `ENABLE_MODULE_IRRLICHT` and configure button
- `Chrono::vehicle` module: click `ENABLE_MODULE_VEHICLE` and configure button
- `Chrono::synchrono` module: click `ENABLE_MODULE_SYNCHRONO` and configure button
- `Chrono::python` module: click `ENABLE_MODULE_PYTHON` and configure button, then set entries as below; Change `<YOUR_MINICONDA3_DIR>` to your real conda path!
```
#NUMPY_INCLUDE_DIR:
<YOUR_MINICONDA3_DIR>/envs/chrono9/lib/python3.9/site-packages/numpy/core/include
```
If meet cmake error for `PYTHON_LIBRARY(ADVANCED)`, exit cmake gui and in the terminal use below:
```
cmake -DPYTHON_EXECUTABLE=<YOUR_MINICONDA3_DIR>/envs/chrono9/bin/python -DPYTHON_LIBRARY=<YOUR_MINICONDA3_DIR>/envs/chrono9/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=<YOUR_MINICONDA3_DIR>/envs/chrono9/include/python3.9 ../chrono
```
- `Chrono::parsers` module: In the terminal open cmake gui again; click `ENABLE_MODULE_PARSERS` and configure button, then set entries as below
```
#urdfdom_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/urdf/lib/urdfdom/cmake
#urdfdom_headers_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/urdf/lib/urdfdom_headers/cmake
#console_bridge_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/urdf/lib/console_bridge/cmake
#tinyxml2_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/urdf/CMake
```
- `Chrono::multicore` module:

Download `blaze-3.8` from [here](https://bitbucket.org/blaze-lib/blaze/src/master/) in folder `<YOUR_HOME_DIR>/Documents/chrono/libraries`;
Click `ENABLE_MODULE_MULTICORE` and configure button, then set entries as below
```
#BLAZE_INSTALL_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/blaze-3.8
```
- `Chrono::opengl` module: click `ENABLE_MODULE_OPENGL` and configure button, then set entries as below
```
#GLM_INCLUDE_DIR:
<YOUR_HOME_DIR>/Documents/chrono/libraries/gl/include/glm
```
- `Chrono::gpu` module: click `ENABLE_MODULE_GPU` and configure button

After choosing all above modules, click **configure** and **generate** buttons. Build files are now available in the `chrono_build` directory. More details can visit [here](https://api.projectchrono.org/tutorial_install_chrono.html).

7. Linux/make:

Depending on the Unix Makefiles generator used during CMake configuration, invoke the appropriate build command
```
make -j 10
```
to build with Make using 10 parallel build threads.

8. Update .bashrc
```
echo 'export LD_LIBRARY_PATH=$HOME/Documents/chrono/libraries/urdf/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CHRONO_DATA_DIR=$HOME/Documents/verti_bench/envs/data/' >> ~/.bashrc
echo 'export PYTHONPATH=$HOME/Documents/chrono_build/bin/' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:$HOME/Documents/' >> ~/.bashrc
```
9. Install ros2 humble related pkgs for systems:
```
sudo apt install ros-humble-grid-map-msgs ros-humble-geometry-msgs
```
10. Test Chrono Build:

Get into the demos folder `<YOUR_HOME_DIR>/Documents/chrono/src/demos/python/` and **activate conda virtual env**, if you have no problems for `core`, `irrlicht`, `robot`, `sensor` and `vehicle` modules, you have successfully built chrono!

# Ten off-road mobility systems
In the Verti-Bench, all task configurations have been documented in YAML files, which can be used to replicate and expand paper's evaluations. Additionally, we provide our terrain generation pipeline, allowing researchers to extend or customize environmental parameters according to their specific research objectives and experimental requirements.

### Prebuilt envs from YAML
All prebuilt envs params and obstacle maps are stored in the folder `envs/data/BenchMaps/sampled_maps/Configs/Final`. For geometry, we use real-world off-road terrain composed of boulders and rocks to create 100 elevation maps under `envs/data/BenchMaps/sampled_maps/Worlds`.

### Customize envs
In the folder `envs/utils`, we have provided SWAE model called **BenchGen.pth** to learn the distribution of real-world vertically challenging terrain to maintain realism. Below are code pipelines to customize Verti-Bench envs.

- `SWAE_gen.ipynb`: Researchers can utilize this model to generate millions of different terrain geometry as they wanted.
- `preHeight-VertiBench.py`: Utilize single terrain patch to query mesh height and save as .npy following BMP coordinate before generating customed env's params.  
- `gen-VertiBench.py`: Generate and store params for rigid and deformable terrain commonly encountered in off-road environments with realistic physical properties under `envs/data/BenchMaps/sampled_maps/Configs/Custom`. To visualize the simulation results, change code to `run_simulation(render=True)`.

### Ten Systems in Verti-Bench
Run our provided ten mobility systems (no task has been used for training) in the 1000 Verti-Bench navigation tasks with prebuilt envs from YAML.

Full Version:
```
python setup.py vehicle=hmmwv system=pid speed=4.0 \ 
world_id=1 scale_factor=1.0 max_time=60 num_experiments=1 \ 
render=true use_gui=false
```
- system: pid (default), eh, mppi, rl, mcl, acl, wmvct, mppi6, tal, tnt
- world_id: from 1 (default) to 100; each world has 10 start and goal pairs
- vehicle: hmmwv (default)
- scale_factor: 1.0 (default), 1/6, 1/10

Simplified Version:
```
python setup.py
```
This is the visualization of HMMWV car on deformable snow terrain with obstacles. When in the Irrlicht GUI, you can press `I` to toggle tool box.

<p align="center">
    <img src="assets/DeformCar.gif" height=350>
</p>

### RL in Verti-Bench
- `off_road_VertiBench.py`: gymnasium environment wrapper to enable RL training
- train: train the models for each example env with stable-baselines3
- test: test scripts to visualize the training environment and debug it
- evaluate: evaluate a trained model

# Citation
If you find our work useful, please consider citing us! 

```bibtex
@article{xu2025verti,
  title={Verti-Bench: A General and Scalable Off-Road Mobility Benchmark for Vertically Challenging Terrain},
  author={Xu, Tong and Pan, Chenhui and Rao, Madhan B and Datar, Aniket and Pokhrel, Anuj and Lu, Yuanjie and Xiao, Xuesu},
  journal={arXiv e-prints},
  pages={arXiv--2502},
  year={2025}
}
```
