using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;

namespace KelpNetTester.Tests
{
    //Learning of MNIST (Handwritten Characters) by MLP
    class Test4
    {
        //Number of mini batches
        const int BATCH_DATA_COUNT = 20;

        //Number of exercises per generation
        const int TRAIN_DATA_COUNT = 3000; //= 60000/20

        //Number of data at performance evaluation
        const int TEST_DATA_COUNT = 200;


        public static void Run()
        {
            //Prepare MNIST data
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //Writing the network configuration in FunctionStack
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, 1024, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(1024, 10, name: "l2 Linear")
            );

            //Declare optimizer
            nn.SetOptimizer(new MomentumSGD());

            //Three generations learning
            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //Total error in the whole
                Real totalLoss = 0;
                long totalLossCount = 0;

                //How many times to run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {

                    //Get data randomly from training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT);

                    //Execute batch learning in parallel
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss = sumLoss;
                    totalLossCount++;

                    //Test the accuracy if you move the batch 20 times
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        //Result output
                        Console.WriteLine("total loss " + totalLoss / totalLossCount);
                        Console.WriteLine("local loss " + sumLoss);

                        Console.WriteLine("\nTesting...");

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
