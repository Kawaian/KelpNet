using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Normalization;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;

namespace KelpNetTester.Tests
{
    //Learning of MNIST (handwritten character) by 15-layer MLP using batch normalization
    //Reference: http: takatakamanbou.hatenablog.com/entry/2015/12/20/233232
    class Test7
    {
        //Number of mini batches
        const int BATCH_DATA_COUNT = 128;

        //Number of exercises per generation
        const int TRAIN_DATA_COUNT = 50000;

        //Number of data at performance evaluation
        const int TEST_DATA_COUNT = 200;

        //Number of middle layers
        const int N = 30; //It operates at 1000 similar to the reference link but it is slow at the CPU

        public static void Run()
        {
            //Prepare MNIST data
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();

            Console.WriteLine("Training Start...");

            //Writing the network configuration in FunctionStack
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, N, name: "l1 Linear"), //L1
                new BatchNormalization(N, name: "l1 BatchNorm"),
                new ReLU(name: "l1 ReLU"),
                new Linear(N, N, name: "l2 Linear"), //L2
                new BatchNormalization(N, name: "l2 BatchNorm"),
                new ReLU(name: "l2 ReLU"),
                new Linear(N, N, name: "l3 Linear"), //L3
                new BatchNormalization(N, name: "l3 BatchNorm"),
                new ReLU(name: "l3 ReLU"),
                new Linear(N, N, name: "l4 Linear"), //L4
                new BatchNormalization(N, name: "l4 BatchNorm"),
                new ReLU(name: "l4 ReLU"),
                new Linear(N, N, name: "l5 Linear"), //L5
                new BatchNormalization(N, name: "l5 BatchNorm"),
                new ReLU(name: "l5 ReLU"),
                new Linear(N, N, name: "l6 Linear"), //L6
                new BatchNormalization(N, name: "l6 BatchNorm"),
                new ReLU(name: "l6 ReLU"),
                new Linear(N, N, name: "l7 Linear"), //L7
                new BatchNormalization(N, name: "l7 BatchNorm"),
                new ReLU(name: "l7 ReLU"),
                new Linear(N, N, name: "l8 Linear"), //L8
                new BatchNormalization(N, name: "l8 BatchNorm"),
                new ReLU(name: "l8 ReLU"),
                new Linear(N, N, name: "l9 Linear"), //L9
                new BatchNormalization(N, name: "l9 BatchNorm"),
                new ReLU(name: "l9 ReLU"),
                new Linear(N, N, name: "l10 Linear"), //L10
                new BatchNormalization(N, name: "l10 BatchNorm"),
                new ReLU(name: "l10 ReLU"),
                new Linear(N, N, name: "l11 Linear"), //L11
                new BatchNormalization(N, name: "l11 BatchNorm"),
                new ReLU(name: "l11 ReLU"),
                new Linear(N, N, name: "l12 Linear"), //L12
                new BatchNormalization(N, name: "l12 BatchNorm"),
                new ReLU(name: "l12 ReLU"),
                new Linear(N, N, name: "l13 Linear"), //L13
                new BatchNormalization(N, name: "l13 BatchNorm"),
                new ReLU(name: "l13 ReLU"),
                new Linear(N, N, name: "l14 Linear"), //L14
                new BatchNormalization(N, name: "l14 BatchNorm"),
                new ReLU(name: "l14 ReLU"),
                new Linear(N, 10, name: "l15 Linear") //L15
            );

            //In this configuration, learning does not proceed
            //FunctionStack nn = new FunctionStack (
            //    new Linear(28 * 28, N), //L1
            //new ReLU (),
            //    new Linear(N, N), //L2
            //new ReLU (),
            //,
            //(Somewhat)
            //,
            //new Linear (N, N), L 14
            //new ReLU (),
            //    new Linear(N, 10) //L15
            //);

            //Declare optimizer
            nn.SetOptimizer(new AdaGrad());

            //Three generations learning
            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //Total error in the whole
                //List <Real> totalLoss = new List <Real> ();
                Real totalLoss = 0;
                long totalLossCounter = 0;

                //How many times to run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //Get data randomly from training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT);

                    //Perform learning
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCounter++;

                    //Test the accuracy if you move the batch 20 times
                    if (i % 20 == 0)
                    {
                        //Result output
                        Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        Console.WriteLine("total loss " + totalLoss / totalLossCounter);
                        Console.WriteLine("local loss " + sumLoss);
                        Console.WriteLine("");
                        Console.WriteLine("Testing...");

                        //Get data randomly from test data
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT);

                        //Run test
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
