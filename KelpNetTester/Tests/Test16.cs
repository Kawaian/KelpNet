using System;
using ChainerModelLoader;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    class Test16
    {
        private const string MODEL_FILE_PATH = "Data/ChainerModel.npz";

        public static void Run()
        {
            //Write the configuration of the network you want to read into FunctionStack and adjust the parameters of each function
            //Make sure to match name to the variable name of Chainer

            FunctionStack nn = new FunctionStack(
                new Convolution2D(1, 2, 3, name: "conv1", gpuEnable: true),//Do not forget the GPU flag if necessary
                new ReLU(),
                new MaxPooling(2, 2),
                new Convolution2D(2, 2, 2, name: "conv2", gpuEnable: true),
                new ReLU(),
                new MaxPooling(2, 2),
                new Linear(8, 2, name: "fl3"),
                new ReLU(),
                new Linear(2, 2, name: "fl4")
            );

            /* Declaration in Chainer
              class NN (chainer.Chain):
                  def __init __ (self):
                      super (NN, self).__ init __ (
                          conv 1 = L. Convolution 2 D (1, 2, 3),
                          conv 2 = L. Convolution 2 D (2, 2, 2),
                          fl3 = L. Linear (8, 2),
                          fl4 = L. Linear (2, 2)
                      )

                  def __call __ (self, x):
                      h_conv 1 = F.relu (self.conv 1 (x))
                      h_pool 1 = F.max_pooling - 2 d (h_conv 1, 2)
                      h_conv 2 = F.relu (self.conv 2 (h_pool 1))
                      h_pool 2 = F.max_pooling - 2 d (h_conv 2, 2)
                      h_fc1 = F.relu (self.fl3 (h_pool2))
                      y = self.fl 4 (h_fc 1)
                      return y
              */


            //Read parameters
            ChainerModelDataLoader.ModelLoad(MODEL_FILE_PATH, nn);

            //Use it as usual
            nn.SetOptimizer(new SGD());

            //Input data
            NdArray x = new NdArray(new Real[,,]{{
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.3, 0.3, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
            }});

            //Teacher signal
            Real[] t = { 0.0, 1.0 };

            //Training conducted
            Trainer.Train(nn, x, t, new MeanSquaredError(), false);

            //Evacuate for results display
            Convolution2D l2 = (Convolution2D)nn.Functions[0];


            //When updating is executed grad will be consumed, so output the value first
            Console.WriteLine("gw1");
            Console.WriteLine(l2.Weight.ToString("Grad"));

            Console.WriteLine("gb1");
            Console.WriteLine(l2.Bias.ToString("Grad"));

            //update
            nn.Update();

            Console.WriteLine("w1");
            Console.WriteLine(l2.Weight);

            Console.WriteLine("b1");
            Console.WriteLine(l2.Bias);
        }
    }
}
