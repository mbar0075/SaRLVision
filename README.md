# Final Year Project (RL Vision)


## Installation
To get started, clone the repository and navigate to it:
```bash
git clone https://github.com/mbar0075/SaRLVision.git
cd SaRLVision
```

You can also clone the environment used for this project using the `environment.yml` file provided in the `Requirements` directory. To do so, you will need to have Anaconda installed on your machine. If you don't have Anaconda installed, you can download it from [here](https://www.anaconda.com/products/distribution). Once you have Anaconda installed, you can run the following commands to install the environment and activate it

To install the environment, run the following command:
```bash
cd Requirements
conda env create -f environment.yml
conda activate SaRLVision
```

Alternatively you can create the environment manually by running the following commands and install the packages in the `requirements.txt` file in the `Requirements` directory:
```bash
cd Requirements
conda env create SaRLVision
conda activate SaRLVision
pip install -r requirements.txt
```

In case you want to install the packages manually, you can do so by running the following commands:
<details>
<summary  style="color: lightblue;"><b> pip install . . .</b></summary>

```bash
conda install swig
conda install nomkl
pip install gymnasium[all]
pip install ufal.pybox2d
pip install pygame
pip install renderlab
pip install numpy
pip install matplotlib
pip install pandas
pip install seaborn
pip install scikit-learn
pip install pycotools

# Installing pytorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118/torch_stable.html

# Installing tensorflow with CUDA 11.2
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"
# Verify the installation:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

In case of any further issues, you can install `cuda` from the following links: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive),
[Windows 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local),
and install the corresponding `pytorch` and `tensorflow` versions from the following links: [PyTorch](https://pytorch.org/get-started/locally/), [TensorFlow](https://www.tensorflow.org/install/pip), respectively.
</details>