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
    //Incorporate label information into teacher signal cDNI
    //Execution of all layers decoupled expressed as model D asynchronously
    //http: ralo23.hatenablog.com/entry/2016/10/22/233405
    class Test12
    {
        //Number of mini batches
        const int BATCH_DATA_COUNT = 256;

        //Number of exercises per generation
        const int TRAIN_DATA_COUNT = 234; //= 60,000 / 256

        //Number of data at performance evaluation
        const int TEST_DATA_COUNT = 100;

        class ResultDataSet
        {
            public readonly NdArray[] Result;
            public readonly NdArray Label;

            public ResultDataSet(NdArray[] result, NdArray label)
            {
                this.Result = result;
                this.Label = label;
            }

            public NdArray GetTrainData()
            {
                Real[] tmp = new Real[(256 + 10) * BATCH_DATA_COUNT];

                for (int i = 0; i < BATCH_DATA_COUNT; i++)
                {
                    tmp[256 + (int)this.Label.Data[i * this.Label.Length] + i * (256 + 10)] = 1;
                    Array.Copy(this.Result[0].Data, i * this.Result.Length, tmp, i * (256 + 10), 256);
                }

                return NdArray.Convert(tmp, new[] { 256 + 10 }, BATCH_DATA_COUNT);
            }
        }

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

            FunctionStack cDNI1 = new FunctionStack(
                new Linear(256 + 10, 1024, name: "cDNI1 Linear1"),
                new BatchNormalization(1024, name: "cDNI1 Nrom1"),
                new ReLU(name: "cDNI1 ReLU1"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack cDNI2 = new FunctionStack(
                new Linear(256 + 10, 1024, name: "cDNI2 Linear1"),
                new BatchNormalization(1024, name: "cDNI2 Nrom1"),
                new ReLU(name: "cDNI2 ReLU1"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "cDNI2 Linear3")
            );

            FunctionStack cDNI3 = new FunctionStack(
                new Linear(256 + 10, 1024, name: "cDNI3 Linear1"),
                new BatchNormalization(1024, name: "cDNI3 Nrom1"),
                new ReLU(name: "cDNI3 ReLU1"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "cDNI3 Linear3")
            );

            //Declare optimizer
            Layer1.SetOptimizer(new Adam(0.00003f));
            Layer2.SetOptimizer(new Adam(0.00003f));
            Layer3.SetOptimizer(new Adam(0.00003f));
            Layer4.SetOptimizer(new Adam(0.00003f));

            cDNI1.SetOptimizer(new Adam(0.00003f));
            cDNI2.SetOptimizer(new Adam(0.00003f));
            cDNI3.SetOptimizer(new Adam(0.00003f));

            for (int epoch = 0; epoch < 10; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //Total error in the whole
                Real totalLoss = 0;
                Real cDNI1totalLoss = 0;
                Real cDNI2totalLoss = 0;
                Real cDNI3totalLoss = 0;

                long totalLossCount = 0;
                long cDNI1totalLossCount = 0;
                long cDNI2totalLossCount = 0;
                long cDNI3totalLossCount = 0;


                //How many times to run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //Get data randomly from training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT);

                    //Run first tier
                    NdArray[] layer1ForwardResult = Layer1.OnForward(datasetX.Data);
                    ResultDataSet layer1ResultDataSet = new ResultDataSet(layer1ForwardResult, datasetX.Label);

                    //Get the inclination of the first layer
                    NdArray[] cDNI1Result = cDNI1.OnForward(layer1ResultDataSet.GetTrainData());

                    //Apply the inclination of the first layer
                    layer1ForwardResult[0].Grad = cDNI1Result[0].Data.ToArray();

                    //Update first layer
                    Layer1.OnBackward(layer1ForwardResult);
                    layer1ForwardResult[0].ParentFunc = null;
                    Layer1.Update();

                    //Run Layer 2
                    NdArray[] layer2ForwardResult = Layer2.OnForward(layer1ResultDataSet.Result);
                    ResultDataSet layer2ResultDataSet = new ResultDataSet(layer2ForwardResult, layer1ResultDataSet.Label);

                    //Get inclination of second layer
                    NdArray[] cDNI2Result = cDNI2.OnForward(layer2ResultDataSet.GetTrainData());

                    //Apply the inclination of the second layer
                    layer2ForwardResult[0].Grad = cDNI2Result[0].Data.ToArray();

                    //Update 2nd tier
                    Layer2.OnBackward(layer2ForwardResult);
                    layer2ForwardResult[0].ParentFunc = null;


                    //Perform learning of first layer cDNI
                    Real cDNI1loss = new MeanSquaredError().Evaluate(cDNI1Result, new NdArray(layer1ResultDataSet.Result[0].Grad, cDNI1Result[0].Shape, cDNI1Result[0].BatchCount));

                    Layer2.Update();

                    cDNI1.OnBackward(cDNI1Result);
                    cDNI1.Update();

                    cDNI1totalLoss += cDNI1loss;
                    cDNI1totalLossCount++;

                    //Run Third Tier
                    NdArray[] layer3ForwardResult = Layer3.OnForward(layer2ResultDataSet.Result);
                    ResultDataSet layer3ResultDataSet = new ResultDataSet(layer3ForwardResult, layer2ResultDataSet.Label);

                    //Get the inclination of the third layer
                    NdArray[] cDNI3Result = cDNI3.OnForward(layer3ResultDataSet.GetTrainData());

                    //Apply the inclination of the third layer
                    layer3ForwardResult[0].Grad = cDNI3Result[0].Data.ToArray();

                    //Update third layer
                    Layer3.OnBackward(layer3ForwardResult);
                    layer3ForwardResult[0].ParentFunc = null;

                    //Perform learning of cDNI for layer 2
                    Real cDNI2loss = new MeanSquaredError().Evaluate(cDNI2Result, new NdArray(layer2ResultDataSet.Result[0].Grad, cDNI2Result[0].Shape, cDNI2Result[0].BatchCount));

                    Layer3.Update();

                    cDNI2.OnBackward(cDNI2Result);
                    cDNI2.Update();

                    cDNI2totalLoss += cDNI2loss;
                    cDNI2totalLossCount++;

                    //Run Layer 4
                    NdArray[] layer4ForwardResult = Layer4.OnForward(layer3ResultDataSet.Result);

                    //Get inclination of the fourth layer
                    Real sumLoss = new SoftmaxCrossEntropy().Evaluate(layer4ForwardResult, layer3ResultDataSet.Label);

                    //Update fourth layer
                    Layer4.OnBackward(layer4ForwardResult);
                    layer4ForwardResult[0].ParentFunc = null;

                    totalLoss += sumLoss;
                    totalLossCount++;

                    //Perform learning of cDNI for the third layer
                    Real cDNI3loss = new MeanSquaredError().Evaluate(cDNI3Result, new NdArray(layer3ResultDataSet.Result[0].Grad, cDNI3Result[0].Shape, cDNI3Result[0].BatchCount));

                    Layer4.Update();

                    cDNI3.OnBackward(cDNI3Result);
                    cDNI3.Update();

                    cDNI3totalLoss += cDNI3loss;
                    cDNI3totalLossCount++;

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //Result output
                    Console.WriteLine("total loss " + totalLoss / totalLossCount);
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\ncDNI1 total loss " + cDNI1totalLoss / cDNI1totalLossCount);
                    Console.WriteLine("cDNI2 total loss " + cDNI2totalLoss / cDNI2totalLossCount);
                    Console.WriteLine("cDNI3 total loss " + cDNI3totalLoss / cDNI3totalLossCount);

                    Console.WriteLine("\ncDNI1 local loss " + cDNI1loss);
                    Console.WriteLine("cDNI2 local loss " + cDNI2loss);
                    Console.WriteLine("cDNI3 local loss " + cDNI3loss);

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
