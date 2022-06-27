[README](./README.md)
====

Official repository for *Multi-Instance Embedding Learning with Deconfounded Instance-Level Prediction (MEDI)*.<br>
Any question can contact with inki.yinji@gmail.com
My home pages:
  * [https://inkiyinji.top](https://inkiyinji.top "Inki's home")
  * [https://inkiyinji.blog.csdn.net](https://inkiyinji.blog.csdn.net "Inki's blog")
  * [https://www.kaggle.com/inkiyinji](https://www.kaggle.com/inkiyinji "Inki's kaggle")

****

# How to use

Just run Main.py

# The file details
* [B2B.py](./B2B.py): The distance function.
* [BagLoader.py](./BagLoader.py): The generator for mnist and fanshionmnist data sets.
* [Classifier.py](./Classifier.py): The instance-level classifier, such as SVM and $k$NN.
* [MIL.py](./MIL.py): The prototype of multi-instance learning (MIL), which can generate improtant variables, e.g., the number of bags $N$, the number of instances $n$.
* [Main.py](./Main.py): The main function for MEDI.
* [NN.py](./NN.py): The optimizer for MEDI based on the attention mechanism.
* [utils.py](./utils.py): Some basic functions.

# The third party library
> numpy, pandas, torch, sklearn, scipy

# The Python version
> 3.9.2

# Some parameters for experiments
* po_label: The main class for generator. For example, if po_label = 0 and data_type="mnist" for MnistLoader, the fashionmnist0 data set (data_space) will be used.
* file_name:
    - If bag_space == None: The data set under the specified path will be used.
    - If bag_space != None: This variable is just used to print the file_name, and the data in bag_sapce will be used.
* epoch: The epoches for optimizer.
* loops: The loops times the k-cv.
* Others:
    - lr: The learning rate.
    - max_dim: The dimension of embedding vector of the bag.
    - norm_type: The norm type for loss function.
    - distill_type: The distill type for embedding function.
                 
# Citation
You can cite our paper as:
```
@article{Zhang:2022:multi,
author		=	{Yu-Xuan Zhangand Mei Yang and Zheng Chun Zhou and Fan Min},
title		=	{Multi-instance embedding learning with deconfounded instance-level prediction},
journal		=	{Research Square},
year		=	{2022},
doi			=	{10.21203/rs.3.rs-1729204/v1},
url			=	{https://www.researchsquare.com/article/rs-1729204/v1}
}
```
