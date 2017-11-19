using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    ///<summary>
    ///Learning XOR by MLP
    ///</summary>
    class Test1
    {
        public static void Run()
        {
            //Number of exercises
            const int learningCount = 10000;

            //Training data
            Real[][] trainData =
            {
                new Real[] { 0, 0 },
                new Real[] { 1, 0 },
                new Real[] { 0, 1 },
                new Real[] { 1, 1 }
            };

            //Training data label
            Real[][] trainLabel =
            {
                new Real[] { 0 },
                new Real[] { 1 },
                new Real[] { 1 },
                new Real[] { 0 }
            };

            //Network configuration is written in FunctionStack
            FunctionStack nn = new FunctionStack(
                new Linear(2, 2, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(2, 2, name: "l2 Linear")
            );

            nn.SetOptimizer(new MomentumSGD());

            //Training looP
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                for (int j = 0; j < trainData.Length; j++)
                {
                    //Describe the loss function at training execution
                    Trainer.Train(nn, trainData[j], trainLabel[j], new SoftmaxCrossEntropy());
                }
            }

            //Show training results
            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray result = nn.Predict(input)[0];
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }

            //Save network after learning
            Console.WriteLine("Model Saveing...");
            ModelIO.Save(nn, "test.nn");

            //Load the network after learning
            Console.WriteLine("Model Loading...");
            FunctionStack testnn = ModelIO.Load("test.nn");

            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray result = testnn.Predict(input)[0];
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }
        }
    }
}
