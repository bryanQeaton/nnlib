# NNlib - A Simple Header-based DNN Library

### Todo:
- Implement early-stopping with patience
- Graph based model initialization for multi-input/multi-output models
- More activation functions
- Choice of SGD, RMSprop, or ADAM optimizers
- Custom loss (?)
- Custom Activations (?)

### Features:
- Saving/Loading models via Cereal-1.3.6
- Sequential model initialization
- Nesterov Momentum + SGD
- Lasso and Ridge regression
- Learning rate scheduling
- Momentum decay scheduling
- Optional RMSnorm
- Global norm-based gradient clipping
- Optional per-layer bias
- User defined training (full control of gradient computing and updating)
- MSE_loss, MAE_loss, Crossentropy_loss, Binary Crossentropy_loss, KL divergence_loss
- Linear, Sigmoid, Softmax, Relu, and Surrogate Relu
- Matplot++ continuous loss plotting

> Design paradigm:
> The model accepts one datapoint at a time
> giving the user full control of the dataset.
> Training is fully external to the model
> allowing for custom training pipelines!


## Simple Use Case:

```c++
#include <vector>
#include "nnlib.h"

int main() {
    auto model=nn::Model(
        32, //input layer size
        { //model architecture: layer size, activation, use_bias
            {256,nn::Relu,false},
            {64,nn::Relu,false},
            {1,nn::Linear,true}
        },
        nn::MSE, //model loss
        0.0001f, //lasso regression term
        0.001f, //ridge regression term
        true //use RMSnorm
        );
        nn::train(
            model,data_x,data_y,1024,.1f, //model, dataset, dataset size, validation split
            250,32, 2,true, //epochs, batch size, plot granularity, use continious plotting
            .001,.0001,.6,.6 //learning rate start and end, momentum decay start and end
            );
            
    return 0;
}
```
### Sources:
[ADAM]: A Method for Stochastic Optimization

[DEMON]: Improved Neural Network Training with Momentum Decay

[sReLU]: The Resurrection of the ReLU

[sADAM]: Bias Correction Debunked

[RMSnorm]: Root Mean Square Layer Normalization

[Nesterov]: Accelerated Gradient and Momentum as Approximations to Regularized Update Descent

[GRADclip]: Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity

[LASSO]: Regression Shrinkage and Selection via the Lasso

[RIDGE]: Biased Estimation for Nonorthogonal Problems


[ADAM]: <https://arxiv.org/pdf/1412.6980>
[DEMON]: <https://arxiv.org/pdf/1910.04952>
[sReLU]: <https://arxiv.org/pdf/2505.22074>
[sADAM]: <https://www.arxiv.org/pdf/2511.20516>
[RMSnorm]: <https://arxiv.org/pdf/1910.07467>
[Nesterov]: <https://arxiv.org/pdf/1607.01981>
[GRADclip]: <https://arxiv.org/pdf/1905.11881>
[LASSO]: <https://webdoc.agsci.colostate.edu/koontz/arec-econ535/papers/Tibshirani%20(JRSS-B%201996).pdf>
[RIDGE]: <https://homepages.math.uic.edu/~lreyzin/papers/ridge.pdf>




Made with <https://dillinger.io/>
