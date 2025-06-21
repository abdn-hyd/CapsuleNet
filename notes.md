## 1. Important Features of Capsule Netowrk

When using a capsule network to process an image, it basically try to inverse the rendering process of an image to obtain important parameters, which can be used to learn for classification task, reconstruction task or just simply encode the information for downstreanm tasks.

The capsule network preserve the important pose and location of objects within the image which called equivariance. If the image rotates for a bit, the orientaion of the specific capsule with highest probability will also change.

Every capsule in the first layer try to predict the output of every capsule in the next layer. The lower layer capsules will try to give the parameters of some parts of the objects, and different parts can finally build up the objects. The output of each capsule in the lower layer will be routing by the agreements.

The routing algorithm is basically an iterative procedure, we initialize the routing weights b_ij to 0. Then, we calculate the weighted sum of predictions of next layer and squash them. Finally, we add dot product(similarity) to b_ij to adjust the weights, which apply the agreements to the weights.
