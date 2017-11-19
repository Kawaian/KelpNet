# KelpNet
KelpNet is a deep learning library implemented in C#

```java
/* SampleCode */
FunctionStack nn = new FunctionStack(
    new Convolution2D(1, 32, 5, pad: 2, name: "l1 Conv2D"),
    new ReLU(name: "l1 ReLU"),
    new MaxPooling(2, 2, name: "l1 MaxPooling"),
    new Convolution2D(32, 64, 5, pad: 2, name: "l2 Conv2D"),
    new ReLU(name: "l2 ReLU"),
    new MaxPooling(2, 2, name: "l2 MaxPooling"),
    new Linear(7 * 7 * 64, 1024, name: "l3 Linear"),
    new ReLU(name: "l3 ReLU"),
    new Dropout(name: "l3 DropOut"),
    new Linear(1024, 10, name: "l4 Linear")
);
```

## Features
- Since the matrix operation is not dependent on the library, all sources are readable, and it is possible to observe everything as to where and what
- We adopt a coding style which describes Keras and Chainer as stacking functions as adopted
- Since OpenCL is adopted for parallel computation, processing can be parallelized not only by GPU but also various computing devices such as CPU and FPGA
> * In order to use OpenCL, additional installation of the corresponding driver may be necessary
> - Intel CPU GPU: https://software.intel.com/en-us/articles/opencl-drivers
> - AMD made CPU GPU: http://www.amd.com/en-us/solutions/professional/hpc/opencl
> - GPU made by Nvidia: https://developer.nvidia.com/opencl

### Benefits made with C#
- Easy to build development environment, easy to learn even for beginners of programming
- There are plenty of options for visually displaying processing results such as WindowsForm and Unity
- Application development for various platforms such as PCs, mobile phones, embedded devices, etc. can be done

## About this library
The core part of this library is implemented with reference to Chainer.
For that reason most function parameters are the same as Chainer and it is possible to develop with reference to samples for Chainer.


## Contact method
If you have any questions or requests, please register with Issues or contact us via Twitter.
Since it does not matter even if it is delicate, if there are any points you have noticed, please do not hesitate to contact us.

Twitter: https://twitter.com/harujoh

## License
- KelpNet [Apache License 2.0]
- Cloo [MIT License] https://sourceforge.net/projects/cloo/

## Implemented function
- Activations:
　・ELU
　・LeakyReLU
　・ReLU
　・Sigmoid
　・Tanh
　・Softmax
　・Softplus
　・Swish
- Connections:
　・Convolution2D
　・Deconvolution2D
　・EmbedID
　・Linear
　・LSTM
- Poolings:
　・AveragePooling
　・MaxPooling
- LossFunctions:
　・MeanSquaredError
　・SoftmaxCrossEntropy
- Optimizers:
　・AdaDelta
　・AdaGrad
　・Adam
　・MomentumSGD
　・RMSprop
　・SGD
- Normalize:
　・BatchNormalization
　・LRN
- Noise:
　・DropOut
　・StochasticDepth
 
Finally, We hope this library will help someone learn
