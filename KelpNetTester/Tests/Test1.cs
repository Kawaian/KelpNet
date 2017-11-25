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
                new Linear(2, 233333, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(233333, 2, name: "l2 Linear")
            );

            nn.SetOptimizer(new MomentumSGD());

            FunctionStack.SwitchToGPU(nn);

            //Training looP
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                Real loss = 0;
                for (int j = 0; j < trainData.Length; j++)
                {
                    using (var output = (NdArray)trainLabel[j])
                    using (var input = (NdArray)trainData[j])
                    {
                        loss = Trainer.Train(nn, input, output, new SoftmaxCrossEntropy());
                    }
                }
                Console.WriteLine($"Batch {i}/{learningCount} | Loss: {loss}");
            }

            //Show training results
            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray result = nn.Predict(input)[0];
                result.ToCpu();
                int resultIndex = Array.IndexOf(result.Data.AsArray(), result.Data.AsArray().Max());
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
                result.ToCpu();
                int resultIndex = Array.IndexOf(result.Data.AsArray(), result.Data.AsArray().Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }
        }
    }
}
