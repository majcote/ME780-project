# ME780-project

Associated files for my project for ME780 - Autonomous Dirving Perception.

Dataset can be found here:

1. Clone repository, clone recursivly to pull the caffe implementation I used, otherwise place your own in the main directory
2. Download data set and unzip in net/Data/
3. Change file locations in
  * net/Data/train.txt
  * net/Data/test.txt
  * net/Data/val.txt
  * net/Models/segnet_train.prototxt
  * net/Models/segnet_solver.prototxt
4. run `./caffe-segnet-cudnn5/build/tools/caffe train -gpu 0 -solver net/Models/segnet_train.prototxt
