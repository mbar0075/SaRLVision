# SaRLVision

<!--

<p align="right" style="text-align: right;">
  <strong>"A reinforcement learning object detector which leverages saliency ranking."</strong>
</p>
<br>
<p align="left" style="text-align: left;">
  <strong>"A self-explainable detector that provides a fully observable action log."</strong>
</p>

## Example of Output

<p align='center'>
  <img src="Diagrams/gif" alt="Output gif" width="50%" height="auto">
</p>

<p align="justify">

## Abstract
*test*

## System Overview

<p align="justify">

Initially, the system proceeds to generate a saliency ranking heatmap using the input image, emphasising regions of interest. It then takes the most important ranks to create an initial bounding box prediction, which is a key stage in object localisation. This prediction is then fed to the RL environment, where an agent navigates through a series of time steps, repeatedly completing actions to improve the bounding box and precisely pinpointing the object within an image, while also predicting the object class label.
</p>

<p align='center'>
  <img src="Diagrams/Architecture.png" alt="Architecture" width="100%" height="auto">
</p>

## Saliency Ranking

<p align='justify'>

The initial process in the development of the system involves the utilisation of saliency ranking to derive an initial bounding box estimate. Alternatively, users may choose not to employ this technique, resulting in the initial bounding box covering the entirety of the input image, a practice commonly observed in existing literature. Following the acquisition of the Saliency Ranking heatmap from [SaRa](https://github.com/dylanseychell/SaliencyRanking), the first stage of this process entails the extraction of a bounding box that delineates the pertinent image segments. This technique considers a proportion of the highest-ranked areas, with a fixed threshold of 30% and number of iterations set to 1. The generation of these initial bounding boxes is critical due to the fact that it allows for the separation and delineation of prominent regions in the image for further refining utilising RL techniques.
</p>

<p align='center'>
  <img src="Diagrams/SaRa -3D plot.png" alt="SaRa" width="100%" height="auto">
</p>

## Reinforcement Learning

<p align="justify">

In the subsequent phase of the devised pipeline, reinforcement learning is harnessed to accomplish object localisation within the images. To this extent the developed system was built via the [gymnasium](https://gymnasium.farama.org/index.html) API, which facilitated the formulation of the problem as a Markov Decision Process (MDP), inspired from the existing literature. Subsequently, Deep Reinforcement Learning (DRL) techniques were applied to approximate the object detection problem.
</p>

### Action Space

<p align="justify">

Similar to methodologies commonly employed in object localisation tasks, the action set $A$ consists of eight transformations that can be applied to the bounding box, along with one action designated to terminate the search process. These transformations are grouped into four subsets: horizontal and vertical box movement, scale adjustment, and aspect ratio modification. Consequently, the agent has four degrees of freedom to adjust the bounding box $[x_1, y_1, x_2, y_2]$ during interactions with the environment. Additionally, a trigger action is incorporated to indicate successful object localisation by the current box, thereby concluding the ongoing search sequence, and drawing an IoR marker on the detected object.
</p>

<p align='center'>
  <img src="Diagrams/Actions_white.png" alt="Actions" width="100%" height="auto">
</p>

### Deep Q-Network Architecture

<p align="justify">

The DQN architecture, introduced in the presented system, assumes responsibility for decision-making in object localisation. To this extent, the designed architecture draws inspiration from methodologies present in the prevalent literature. Our proposed approach, introduces four DQN variants:
1. `Vanilla DQN (DQN)`
2. `Double DQN (DDQN)`
3. `Dueling DQN (Dueling DQN)`
4. `Double Dueling DQN (D3QN)`

Our approach advocates for a deeper DQN network to bolster decision-making capabilities and enhance learning complexity. To mitigate concerns regarding overfitting, dropout layers are seamlessly integrated into the network architecture. Additionally, this work develops a Dueling DQN Agent to improve learning efficiency by decoupling state and advantage functions. The Dueling DQN design divides the $Q$-value function into two streams, allowing the agent to better comprehend the value of doing specific actions in different situations. The proposed approach also evaluates DDQN and D3QN techniques, which have also not been previously examined, in pursuit of achieving better results.
</p>

<p align='center'>
  <img src="Diagrams/dqn_architectures.png" alt="DQN Architecture" width="70%" height="auto">
</p>

### Self-Explainability

<p align="justify">
The study proposes a system that creates a log and displays the current environment in several rendering modes to illustrate explainability, as demonstrated below:

<p align='center'>
  <img src="Diagrams/Visualisations.png" alt="Visualisations" width="100%" height="auto">
</p>

These visualisations provide users with insights into the current action being performed, the current IoU, the current Recall, the environment step counter, the current reward, and a clear view of the current bounding box and ground truth bounding box locations in the original image. Furthermore, unlike all object detectors and methodologies previously discussed, this methodology permits decision-making observation during the training phase, albeit there is a slight time overhead for the creation of visualisations. 
Nonetheless, the system provides a clear log outlining the framework's decision-making process for current item detection, allowing insight into the object detector's training and assessment, as observed below:

<p align='center'>
  <img src="Diagrams/Self-Explainability.png" alt="Self-Explainability" width="100%" height="auto">
</p>

</p>

## SaRLVision Window

<p align="justify">

The SaRLVision Window provides a real-time view of the object detection process, displaying the current state of the environment, the actions being taken, and the corresponding results. This interactive window is designed to be user-friendly, providing a clear and intuitive interface for users to understand the workings of the system. It also includes controls for pausing, resuming, and stopping the detection process, giving users a degree of control over the system's operation.

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Diagrams/SaRLVisionWindow1.png" alt="Window1"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Diagrams/SaRLVisionWindow2.png" alt="Window2" width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Diagrams/SaRLVisionWindow3.png" alt="Window3" width="100%" height="auto" />
    </td>
  </tr>
</table>
</p>

This window is implemented using the `pygame` library, which is a popular framework for building interactive applications and games in Python. Pygame provides a set of functions and classes for creating graphical user interfaces, handling user input, and rendering graphics on the screen. By utilizing the Pygame API, the SaRLVision Window is able to provide a visually appealing and responsive interface for users to interact with the object detection system.

</p>

## Getting Started

<p align="justify">

The following jupyter notebooks are provided to demonstrate the functionality of the system:
- [Dataset Notebooks](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/Datasets)
- [Saliency Ranking Threshold Experiments](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/Threshold%20Experiments)
- [Training.ipynb](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/RL%20Agent%20Training/Training.ipynb)
- [Evaluation.ipynb](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/RL%20Agent%20Training/Evaluation.ipynb)
- [Testing.ipynb](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/RL%20Agent%20Training/Testing.ipynb)
- [Visualisations.ipynb](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/RL%20Agent%20Training/Visualisations.ipynb)
- [Self-Explainability.ipynb](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/RL%20Agent%20Training/Self-Explainability.ipynb)
- [Plotting Results.ipynb](https://github.com/mbar0075/SaRLVision/tree/main/Experiments/RL%20Agent%20Training/Plotting_Results.ipynb) 
  
</p>
-->

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
<summary  style="color: lightblue; cursor: pointer"><i> pip install . . .</i></summary>

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

</p>