using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //Learning of Sin function by MLP

    //Increasing the number of cycles to be learned or increasing the number of samples (N)
    //It may be better to try as a challenge
    class Test3
    {
        //Learning frequency
        const int EPOCH = 1000;

        //Number of divisions per period
        const int N = 50;

        public static void Run()
        {
            Real[][] trainData = new Real[N][];
            Real[][] trainLabel = new Real[N][];

            for (int i = 0; i < N; i++)
            {
                //Prepare Sin wave for one cycle
                Real radian = -Math.PI + Math.PI * 2.0 * i / (N - 1);
                trainData[i] = new[] { radian };
                trainLabel[i] = new Real[] { Math.Sin(radian) };
            }

            //Writing the network configuration in FunctionStack
            FunctionStack nn = new FunctionStack(
                new Linear(1, 4, name: "l1 Linear"),
                new Tanh(name: "l1 Tanh"),
                new Linear(4, 1, name: "l2 Linear")
            );

            //Declaration of optimizer
            nn.SetOptimizer(new SGD());

            //Training loop
            for (int i = 0; i < EPOCH; i++)
            {
                //For error aggregation
                Real loss = 0;

                for (int j = 0; j < N; j++)
                {
                    //When training is executed in the network, an error is returned to the return value
                    loss += Trainer.Train(nn, trainData[j], trainLabel[j], new MeanSquaredError());
                }

                if (i % (EPOCH / 10) == 0)
                {
                    Console.WriteLine("loss:" + loss / N);
                    Console.WriteLine("");
                }
            }

            //Show training results
            Console.WriteLine("Test Start...");

            foreach (Real[] val in trainData)
            {
                Console.WriteLine(val[0] + ":" + nn.Predict(val)[0].Data[0]);
            }
        }
    }
}
