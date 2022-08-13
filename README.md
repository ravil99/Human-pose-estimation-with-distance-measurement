# Human pose estimation with distance measurement relative to ArUco marker.
##Innopolis University. Computer Vision. Course project. 

## My work was related to ArUco Markers.

## Environment setup
Recomended 
* Python 3.8
* GPU at least with 4Gb memory
* CUDA 11.7 + cuDNN 8.4.1

Environment setup script:
```
python -m venv env
. env/bin/activate
pip install -r requirements.txt
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone https://github.com/isarandi/poseviz.git
cd poseviz
pip install .
cd ../
rm -rf poseviz/
```

To launch project: `python main.py`  
You can change main settings in `config.yaml` file. For example you can turn on or off separate module.

## Project description
It consists of 3 parts:
* ArUco marker detector
* Depth estimation
* 3D and 2D human pose estimator

## ArUco marker detector
The code is well described [here](https://github.com/4ku/Human-pose-estimation-for-robot-collaboration/tree/master/modules/aruco).  
This module detect ArUco markers and calculate distance to them.  
![](modules/aruco/demo.gif)

## Depth estimator
Get depth map from single image. Depth estimator based on [MiDaS](https://github.com/isl-org/MiDaS) approach. Models was taken from the MiDaS [PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/).  

![](img/depth_estimation.png)
![](img/depth_orig.png)

## Pose estimation
Estimates human position in 2D and 3D spaces. Approach was based on [MeTRAbs](https://github.com/isarandi/metrabs) solution. Full documentation is available [here](https://github.com/isarandi/metrabs/tree/master/docs).  

![](img/demo_metrabs.gif)  

## Project demo
Project demo is available on [YouTube](https://youtu.be/4IWl8UEf0FA)  

![](img/human_distance_demo.gif)

## Results
Estimating distance to the human with only one camera is not so precise. So it would be better to use `multiple` cameras for such problem.  

Below you can see results of some experiment. Video of experiment you can see on [Youtube here](https://youtu.be/PDebTES1UxI). The error depends on the depth estimator, so sometimes it can be significantly wrong.

![](img/Distance_to_human.png)
