using System;
using KelpNet.Common;
using KelpNetTester.Benchmarker;
using KelpNetTester.Tests;

namespace KelpNetTester
{
    // Uncomment the test you want to run
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            // Comment out here if you want to run on .Net Framework
            Weaver.Initialize(ComputeDeviceTypes.Gpu);
            // Weaver.Initialize(ComputeDeviceTypes.Cpu, 1); 
            // Subscript required if there are multiple devices

            // learning XOR with MLP
            // Test1.Run();

            // Learning XOR with MLP (Regression version)
            // Test2.Run();

            // Learning of Sin function by MLP
            //Test3.Run();

            // Learning of MNIST (handwritten character) by MLP
            // Test4.Run();

            // Reproduction of Excel CNN
            // Test5.Run();

            // Learning of MNIST with 5 - layer CNN
            // Test6.Run();

            // Learning of MNIST by 15 layer MLP using BatchNorm
            // Test7.Run();

            // Learning of Sin function by LSTM
            // Test8.Run();

            // RNNLM with a simple RNN
            // Test9.Run();

            // RNNLM by LSTM
            // Test10.Run();

            // MNIST learning by Decoupled Neural Interfaces using Synthetic Gradients
            // Test11.Run();

            // Set DNI of Test 11 as cDNI
            // Test12.Run();

            // Test of Deconvolution 2D(Winform)
            // new Test 13 WinForm().ShowDialog();

            // Connect Test 6 and execute
            // Test14.Run();

            // Test that reads VGG 16 of Caffe model and makes image classification
            // Test15.Run();

            // Load the same contents as Test 5 of Chainer model and execute it
            // Test16.Run();

            // Test to load image classification by loading RESNET of Caffe model
            Test17.Run(Test17.ResnetModel.ResNet50);
            // Please choose an arbitrary Resnet model

            // split execution of Linear
            // TestX.Run();

            // benchmark
            //SingleBenchmark.Run();

            Console.WriteLine("Test Done ...");
            Console.Read();
        }
    }
}