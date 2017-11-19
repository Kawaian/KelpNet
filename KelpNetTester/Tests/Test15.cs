using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using CaffemodelLoader;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using TestDataManager;

namespace KelpNetTester.Tests
{
    //Test to read VGA 16 of Caffe model and make image classification
    class Test15
    {
        private const string DOWNLOAD_URL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel";
        private const string MODEL_FILE = "VGG_ILSVRC_16_layers.caffemodel";
        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        public static void Run()
        {
            OpenFileDialog ofd = new OpenFileDialog { Filter = ". Image file (*. Jpg; *. Png; *. Gif; *. Bmp) | *. Jpg; *. Png; *. Gif; *. Bmp | all files (*. *) | *. *" };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                Console.WriteLine("Model Loading.");
                string modelFilePath = InternetFileDownloader.Donwload(DOWNLOAD_URL, MODEL_FILE);
                List<Function> vgg16Net = CaffemodelDataLoader.ModelLoad(modelFilePath);
                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                //Initialize GPU
                for (int i = 0; i < vgg16Net.Count - 1; i++)
                {
                    if (vgg16Net[i] is Convolution2D || vgg16Net[i] is Linear || vgg16Net[i] is MaxPooling)
                    {
                        ((IParallelizable) vgg16Net[i]).SetGpuEnable(true);
                    }
                }

                FunctionStack nn = new FunctionStack(vgg16Net.ToArray());

                //Compress layer
                nn.Compress();

                Console.WriteLine("Model Loading done.");

                do
                {
                    //Before inputting to the network, set the resolution to 224px x 224px x 3ch
                    Bitmap baseImage = new Bitmap(ofd.FileName);
                    Bitmap resultImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);
                    g.DrawImage(baseImage, 0, 0, 224, 224);
                    g.Dispose();

                    Real[] bias = {-123.68, -116.779, -103.939}; //The channel order of the correction value follows the input image
                    NdArray imageArray = NdArrayConverter.Image2NdArray(resultImage, false, true, bias);

                    Console.WriteLine("Start predict.");
                    Stopwatch sw = Stopwatch.StartNew();
                    NdArray result = nn.Predict(imageArray)[0];
                    sw.Stop();

                    Console.WriteLine("Result Time : " +
                                      (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") +
                                      "μs");

                    int maxIndex = Array.IndexOf(result.Data, result.Data.Max());
                    Console.WriteLine("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);
                } while (ofd.ShowDialog() == DialogResult.OK);
            }
        }
    }
}
