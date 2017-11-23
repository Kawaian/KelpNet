using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;

namespace KelpNetTester.Tests
{
    class Test18
    {
        //Number of mini batches
        const int BATCH_DATA_COUNT = 20;

        //Number of exercises per generation
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20

        //Number of data at performance evaluation
        const int TEACH_DATA_COUNT = 200;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();

            //Prepare MNIST data
            Console.WriteLine("CIFAR Data Loading...");
            CifarData cifarData = new CifarData();

            //Writing the network configuration in FunctionStack
            FunctionStack nn = new FunctionStack(
                new Convolution2D(3, 32, 3, name: "l1 Conv2D", gpuEnable: true),
                new ReLU(name: "l1 ReLU"),
                new MaxPooling(2, name: "l1 MaxPooling", gpuEnable: true),
                new Dropout(0.25, name: "l1 DropOut"),
                new Convolution2D(32, 64, 3, name: "l2 Conv2D", gpuEnable: true),
                new ReLU(name: "l2 ReLU"),
                new MaxPooling(2, 2, name: "l2 MaxPooling", gpuEnable: true),
                new Dropout(0.25, name: "l2 DropOut"),
                new Linear(13 * 13 * 64, 512, name: "l3 Linear", gpuEnable: true),
                new ReLU(name: "l3 ReLU"),
                new Dropout(name: "l3 DropOut"),
                new Linear(512, 10, name: "l4 Linear", gpuEnable: true)
            );

            //Declare optimizer
            nn.SetOptimizer(new Adam());

            Console.WriteLine("Training Start...");

            //Three generations learning
            for (int epoch = 1; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + epoch);

                //Total error in the whole
                Real totalLoss = 0;
                long totalLossCount = 0;

                //How many times to run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    sw.Restart();

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);

                    //Get data randomly from training data
                    TestDataSet datasetX = cifarData.GetRandomXSet(BATCH_DATA_COUNT);

                    //Execute batch learning in parallel
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCount++;

                    //Result output
                    Console.WriteLine("total loss " + totalLoss / totalLossCount);
                    Console.WriteLine("local loss " + sumLoss);

                    sw.Stop();
                    Console.WriteLine("time" + sw.Elapsed.TotalMilliseconds);

                    //Test the accuracy if you move the batch 20 times
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nTesting...");

                        //テストデータからランダムにデータを取得
                        TestDataSet datasetY = cifarData.GetRandomYSet(TEACH_DATA_COUNT);

                        //Get data randomly from test data
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
