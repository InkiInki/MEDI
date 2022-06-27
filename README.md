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
* 
