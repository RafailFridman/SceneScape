# SceneScape: Text-Driven Consistent Scene Generation
## [<a href="https://scenescape.github.io/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-SceneScape-b31b1b.svg)](https://arxiv.org/abs/2302.01133)

[//]: # (image is in teaser.png)
![teaser](assets/teaser.png)




https://github.com/RafailFridman/SceneScape/assets/22198039/de89e930-4aba-4149-935a-d293fba793d4




**SceneScape** is a method for  generating videos of indoor scenes walkthroughs from text, as described <a href="https://arxiv.org/abs/2302.01133" target="_blank">here</a>.
### Abstract
>We present a method for text-driven perpetual view generation -- synthesizing long-term videos of various scenes solely, given an input text prompt describing the scene and camera poses. We introduce a novel framework that generates such videos in an online fashion by combining the generative power of a pre-trained text-to-image model with the geometric priors learned by a pre-trained monocular depth prediction model. To tackle the pivotal challenge of achieving 3D consistency, i.e., synthesizing videos that depict geometrically-plausible scenes, we deploy an online test-time training to encourage the predicted depth map of the current frame to be geometrically consistent with the synthesized scene. The depth maps are used to construct a unified mesh representation of the scene, which is progressively constructed along the video generation process. In contrast to previous works, which are applicable only to limited domains, our method generates diverse scenes, such as walkthroughs in spaceships, caves, or ice castles.


## Getting Started
### Installation
For the installation to be done correctly, please proceed only with CUDA-compatible GPU available.

Clone the repo and create the environment:
```
git clone https://github.com/rafailfridman/SceneScape.git
cd SceneScape
conda create --name scenescape python=3.10
conda activate scenescape 
```
We are using  <a href="https://github.com/facebookresearch/pytorch3d" target="_blank">Pytorch3D</a> to perform rendering.
Run the following commands to install it or follow their <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md" target="_blank">installation guide</a> (it may take some time).
```
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
Install the rest of the requirements:
```
pip install -r requirements.txt
```
#### Splatting installation (OPTIONAL)
If you want to test how our method works without the mesh, just by warping pixels from frame `i` to `i+1`, you will have to install <a href="https://github.com/sniklaus/softmax-splatting" target="_blank">softmax-splatting</a> package, following their instructions.
Initialize the submodule:
```
git submodule update --init
```
Then to run with splatting and not the mesh, change the `use_splatting` flag in the `configs/base-config.yaml` file to `True`.

### Run examples 
* **SceneScape** is designed to generate walkthough videos of static scenes. It is not designed for generating dynamic content (e.g. people, cars moving, etc.).
* Running **SceneScape** multiple times with the same inputs can lead to slightly different results.
* Method is **not** designed to generate **outdoor scenes** because of the mesh used as a unified representation.

Method was tested to run on a single Tesla V100 32GB, and takes ~20GB of video memory.
It takes approximately 2.5 hours to generate a 50 frames video on a single Tesla V100 32GB.

Run the following command to start training
```
python run.py --example_config config/example_configs/dungeon.yaml
```

Intermediate results will be saved to `output` during optimization. Once the method is done, the video will be saved to `output/output.mp4`.
You can change the prompt in the `config/example_configs/dungeon.yaml` file to generate different videos, or create your own config. The base parameters of our method are located in `configs/base-config.yaml`.

For more results see the [supplementary material](https://scenescape.github.io/sm/index.html).



## Citation
```
@article{SceneScape,
      author    = {Rafail Fridman and Amit Abecasis and Yoni Kasten and Tali Dekel},
      title     = {SceneScape: Text-Driven Consistent Scene Generation},
      journal   = {arXiv preprint arXiv:2302.01133},
      year      = {2023},
  }
```
