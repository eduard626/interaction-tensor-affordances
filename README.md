# Multiple affordance detection code and data.
This repo contains the implementation of our paper "[Geometric Affordance Perception: Leveraging Deep 3D Saliency With the Interaction Tensor](https://www.frontiersin.org/articles/10.3389/fnbot.2020.00045)", from Eduardo Ruiz and Walterio Mayol-Cuevas at the University of Bristol, UK.

<p align="center">
	<img src="https://media.giphy.com/media/jRYPXFp9Iveo0GqZpK/giphy.gif" alt="photo not available" height="50%">
</p>

If you use our code or method in your work, please cite the following:
```
@article{Ruiz-Mayol2020,
author={Ruiz, Eduardo and Mayol-Cuevas, Walterio},   	 
title={Geometric Affordance Perception: Leveraging Deep 3D Saliency With the Interaction Tensor},      
journal={Frontiers in Neurorobotics},      
volume={14},      
pages={45},     
year={2020},             
doi={10.3389/fnbot.2020.00045},      	
issn={1662-5218},   
``` 

### Contents
1. [Dependencies](https://github.com/eduard626/interaction-tensor-affordances#dependencies)
2. [Data](https://github.com/eduard626/interaction-tensor-affordances#data)
3. [Saliency training](https://github.com/eduard626/interaction-tensor-affordances#saliency-training)
4. [Descriptor computation](https://github.com/eduard626/interaction-tensor-affordances#descriptor-computation)
5. [Multiple affordance prediction](https://github.com/eduard626/interaction-tensor-affordances#multiple-affordance-prediction)
5. [Multiple affordance prediction](https://github.com/eduard626/interaction-tensor-affordances#multiple-affordance-prediction)

#### Dependencies
You can find all the dependencies for _training_ our saliency-based method in the file:

 ```conda_environment.yml```
 
The setup was last tested with Ubuntu 16.04 on June 2020.

#### Data
As per our paper, we train affordance saliency on pointclouds extracted with our method from the ICRA'18 paper, [code here](https://github.com/eduard626/interaction-tensor).
Essentially, we run single affordance prediction on 20 RGB-D scenes, which are a combination of our own captures and publicly
available data.
These pointclouds are available of upon request, but you can also use your own and follow our [previous work](https://github.com/eduard626/interaction-tensor) to produce data.

For testing multiple affordance prediction, we employed a subset of the ScanNet dataset, we used the subset of scenes with the following ids: [scannet_subset](https://github.com/eduard626/deep-interaction-tensor/blob/master/figures/scannet_scenes.txt)
Should you require the same data, please follow the instructions of [ScanNet](http://www.scan-net.org/) project.  

We provide a smaller training/validation dataset (and some auxiliary data) to get you started. This data can be found in the data dir of mPointNet
```
+-- README.md
+-- _mPointNet/
|   +-- _data/
|   |   +-- _new_data_centered/
|   |   |   +-- MultilabelDataSet_splitTest.h5
|   |   |   +-- MultilabelDataSet_splitTrain.h5
|   |   |   +-- ...   
|   +-- log/
|   +-- dump/
```
##### Single-affordance Tensors
The 84 tensors for all the affordances considered for our experiments will be available soon. If want to start building your own, have a look at our [previous work](https://github.com/eduard626/interaction-tensor). You will need CAD models for training the affordances/interactions
of interest.

#### Multiple-affordance multi-label datasets
We here provide a script to generate multiple-affordance datasets from sets of single affordance predictions, method used for our paper.

You need to run single-affordance predictions on an input scene of your choice. For that, you will need single-affordance tensors (from above) and the code from our
previous method, which you can get [here](https://github.com/eduard626/deep-interaction-tensor). The code will store
results on the `data` directory. Once you have computed individual affordance predictions you can run the following script
```
python multiple_affordance_dataset.py 
```
This script will look for results from individual predictions and use that data to create a multi-affordance multi-label dataset suitable for
training saliency. The code will create one small dataset per scene, you can use as many of these small datasets
as you need in order to gather enough data to train saliency, for instance.
```
python create_split.py 90 10
```

This will look for all the _small_ datasets put them together and create a 90/10 split for training and validation.

#### Saliency training

Saliency training is based on the PointNet++ architecture with the modifications discussed in the paper.
The code for this stage of our method can be found in the ```mPointNet``` directory. Please refer to the original [PointNet++](https://github.com/charlesq34/pointnet2) instructions
on how to compile the tf_operators, you should be fine compiling and running everything with the provided conda environment from above.

In order to train the network you want to run:
```
python train_affordances.py
```
With the provided dataset, this command will train the network with the default configuration as in our paper, e.g. 84 affordances + background, cross-entropy loss with L2-normalization, batch size=32, sampling 1024 points, etc.

In order to evaluate the network and extract scene saliency you need to run:
```
python evaluate_affordances.py
```
Which will write results and useful data into the ```dump/``` directory

#### Descriptor Computation
With the learned saliency and our Interaction Tensors we compute a multiple-affordance descriptor for fast affordance prediction.
The code to compute this descriptor needs the data crated by the previous step and single-affordance interaction tensors (see point 2 above). To compute the descriptor run:
```
python backprojection.py
```
This code projects the saliency learned by network back into the corresponding tensors. The resulting
descriptor will be written to the `data/` directory.

#### Multiple Affordance Prediction
Once the descriptor has been computed, you can carry on and detect multiple affordance at fast rates
with our C++ code. The code for that is the same as in our previous work, the difference being the descriptor that you feed in 
(e.g. the one produced by the previous step). Instructions and code for predictions are found [here](https://github.com/eduard626/deep-interaction-tensor) 


##### Support
If you have questions or problems with the code, please create an issue in this repo.
#### Author
[Eduardo Ruiz](https://ed-ruiz.github.io/)
