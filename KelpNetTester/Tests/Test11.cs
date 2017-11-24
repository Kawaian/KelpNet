using System;
using System.Linq;
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
    //Learning of MNIST (handwritten character) by Decoupled Neural Interfaces using Synthetic Gradients
    //http: ralo23.hatenablog.com/entry/2016/10/22/233405
    class Test11
    {
        //Number of mini batches
        const int BATCH_DATA_COUNT = 200;

        //Number of exercises per generation
        const int TRAIN_DATA_COUNT = 300; //= 60000/20

        //Number of data at performance evaluation
        const int TEST_DATA_COUNT = 1000;

        public static void Run()
        {
            //Prepare MNIST data
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //Writing the network configuration in FunctionStack
            FunctionStack Layer1 = new FunctionStack(
                new Linear(28 * 28, 256, name: "l1 Linear"),
                new BatchNormalization(256, name: "l1 Norm"),
                new ReLU(name: "l1 ReLU")
            );

            FunctionStack Layer2 = new FunctionStack(
                new Linear(256, 256, name: "l2 Linear"),
                new BatchNormalization(256, name: "l2 Norm"),
                new ReLU(name: "l2 ReLU")
            );

            FunctionStack Layer3 = new FunctionStack(
                new Linear(256, 256, name: "l3 Linear"),
                new BatchNormalization(256, name: "l3 Norm"),
                new ReLU(name: "l3 ReLU")
            );

            FunctionStack Layer4 = new FunctionStack(
                new Linear(256, 10, name: "l4 Linear")
            );

            //FunctionStack itself is also stacked as Function
            FunctionStack nn = new FunctionStack
            (
                Layer1,
                Layer2,
                Layer3,
                Layer4
            );

            FunctionStack DNI1 = new FunctionStack(
                new Linear(256, 1024, name: "DNI1 Linear1"),
                new BatchNormalization(1024, name: "DNI1 Nrom1"),
                new ReLU(name: "DNI1 ReLU1"),
                new Linear(1024, 1024, name: "DNI1 Linear2"),
                new BatchNormalization(1024, name: "DNI1 Nrom2"),
                new ReLU(name: "DNI1 ReLU2"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack DNI2 = new FunctionStack(
                new Linear(256, 1024, name: "DNI2 Linear1"),
                new BatchNormalization(1024, name: "DNI2 Nrom1"),
                new ReLU(name: "DNI2 ReLU1"),
                new Linear(1024, 1024, name: "DNI2 Linear2"),
                new BatchNormalization(1024, name: "DNI2 Nrom2"),
                new ReLU(name: "DNI2 ReLU2"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "DNI2 Linear3")
            );

            FunctionStack DNI3 = new FunctionStack(
                new Linear(256, 1024, name: "DNI3 Linear1"),
                new BatchNormalization(1024, name: "DNI3 Nrom1"),
                new ReLU(name: "DNI3 ReLU1"),
                new Linear(1024, 1024, name: "DNI3 Linear2"),
                new BatchNormalization(1024, name: "DNI3 Nrom2"),
                new ReLU(name: "DNI3 ReLU2"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "DNI3 Linear3")
            );

            //Declare optimizer
            Layer1.SetOptimizer(new Adam());
            Layer2.SetOptimizer(new Adam());
            Layer3.SetOptimizer(new Adam());
            Layer4.SetOptimizer(new Adam());

            DNI1.SetOptimizer(new Adam());
            DNI2.SetOptimizer(new Adam());
            DNI3.SetOptimizer(new Adam());

            //Three generations learning
            for (int epoch = 0; epoch < 20; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                Real totalLoss = 0;
                Real DNI1totalLoss = 0;
                Real DNI2totalLoss = 0;
                Real DNI3totalLoss = 0;

                long totalLossCount = 0;
                long DNI1totalLossCount = 0;
                long DNI2totalLossCount = 0;
                long DNI3totalLossCount = 0;

                //How many times to run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //Get data randomly from training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT);

                    //Run first tier
                    NdArray[] layer1ForwardResult = Layer1.Forward(datasetX.Data);

                    //Get the inclination of the first layer
                    NdArray[] DNI1Result = DNI1.Forward(layer1ForwardResult);

                    //Apply the inclination of the first layer
                    layer1ForwardResult[0].Grad = DNI1Result[0].Data.ToArray();

                    //Update first layer
                    Layer1.Backward(layer1ForwardResult);
                    layer1ForwardResult[0].ParentFunc = null; //I ran Backward, so I cut off the calculation graph
                    Layer1.Update();

                    //Run Layer 2
                    NdArray[] layer2ForwardResult = Layer2.Forward(layer1ForwardResult);

                    //Get inclination of second layer
                    NdArray[] DNI2Result = DNI2.Forward(layer2ForwardResult);

                    //Apply the inclination of the second layer
                    layer2ForwardResult[0].Grad = DNI2Result[0].Data.ToArray();

                    //Update 2nd tier
                    Layer2.Backward(layer2ForwardResult);
                    layer2ForwardResult[0].ParentFunc = null;

                    //Perform learning of first layer DNI
                    Real DNI1loss = new MeanSquaredError().Evaluate(DNI1Result, new NdArray(layer1ForwardResult[0].Grad, DNI1Result[0].Shape, DNI1Result[0].BatchCount));

                    Layer2.Update();

                    DNI1.Backward(DNI1Result);
                    DNI1.Update();

                    DNI1totalLoss += DNI1loss;
                    DNI1totalLossCount++;

                    //Run Third Tier
                    NdArray[] layer3ForwardResult = Layer3.Forward(layer2ForwardResult);

                    //Get the inclination of the third layer
                    NdArray[] DNI3Result = DNI3.Forward(layer3ForwardResult);

                    //Apply the inclination of the third layer
                    layer3ForwardResult[0].Grad = DNI3Result[0].Data.ToArray();

                    //Update third layer
                    Layer3.Backward(layer3ForwardResult);
                    layer3ForwardResult[0].ParentFunc = null;

                    //Perform learning of DNI for layer 2
                    Real DNI2loss = new MeanSquaredError().Evaluate(DNI2Result, new NdArray(layer2ForwardResult[0].Grad, DNI2Result[0].Shape, DNI2Result[0].BatchCount));

                    Layer3.Update();

                    DNI2.Backward(DNI2Result);
                    DNI2.Update();

                    DNI2totalLoss += DNI2loss;
                    DNI2totalLossCount++;

                    //Run Layer 4
                    NdArray[] layer4ForwardResult = Layer4.Forward(layer3ForwardResult);

                    //Get inclination of the fourth layer
                    Real sumLoss = new SoftmaxCrossEntropy().Evaluate(layer4ForwardResult, datasetX.Label);

                    //Update fourth layer
                    Layer4.Backward(layer4ForwardResult);
                    layer4ForwardResult[0].ParentFunc = null;

                    totalLoss += sumLoss;
                    totalLossCount++;

                    //Perform DNI learning for layer 3
                    Real DNI3loss = new MeanSquaredError().Evaluate(DNI3Result, new NdArray(layer3ForwardResult[0].Grad, DNI3Result[0].Shape, DNI3Result[0].BatchCount));

                    Layer4.Update();

                    DNI3.Backward(DNI3Result);
                    DNI3.Update();

                    DNI3totalLoss += DNI3loss;
                    DNI3totalLossCount++;

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //Result output
                    Console.WriteLine("total loss " + totalLoss / totalLossCount);
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\nDNI1 total loss " + DNI1totalLoss / DNI1totalLossCount);
                    Console.WriteLine("DNI2 total loss " + DNI2totalLoss / DNI2totalLossCount);
                    Console.WriteLine("DNI3 total loss " + DNI3totalLoss / DNI3totalLossCount);

                    Console.WriteLine("\nDNI1 local loss " + DNI1loss);
                    Console.WriteLine("DNI2 local loss " + DNI2loss);
                    Console.WriteLine("DNI3 local loss " + DNI3loss);

                    //Test the accuracy if you move the batch 20 times
                    if (i % 20 == 0)
                    {
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
