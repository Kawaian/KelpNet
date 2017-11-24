using System;
using KelpNet.Common;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //Based on the image outputted by a certain learned filter, a filter equivalent to the filter is acquired
    //Console version
    //Transplant origin: http: qiita.com/samacoba/items/958c02f455ca5f3a475d
    class Test13
    {
        public static void Run()
        {
            //Create a target filter (If it is practical, here is an unknown value)
            Deconvolution2D decon_core = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true)
            {
                Weight = { Data = MakeOneCore() }
            };

            Deconvolution2D model = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true);

            SGD optimizer = new SGD(learningRate: 0.00005); //When it is big, it diverges.
            model.SetOptimizer(optimizer);
            MeanSquaredError meanSquaredError = new MeanSquaredError();

            //At the transplant source, we are educating with the same educational image, but changing to learning closer to practice
            for (int i = 0; i < 11; i++)
            {
                //Generate random dotted images
                NdArray img_p = getRandomImage();

                //Output a learning image with a target filter
                NdArray[] img_core = decon_core.OnForward(img_p);

                //Output an image with an unlearned filter
                NdArray[] img_y = model.OnForward(img_p);

                Real loss = meanSquaredError.Evaluate(img_y, img_core);

                model.OnBackward(img_y);
                model.Update();

                Console.WriteLine("epoch" + i + " : " + loss);
            }
        }

        static NdArray getRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            //Randomly make 0.1% points
            Real[] img_p = new Real[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother.Dice.Next(0, 1000);
                img_p[i] = img_p[i] > 999 ? 0 : 1;
            }

            return new NdArray(img_p, new[] { N, img_h, img_w }, 1);
        }

        //Create one spherical pattern (Gauss)
        static Real[] MakeOneCore()
        {
            int max_xy = 15;
            Real sig = 5;
            Real sig2 = sig * sig;
            Real c_xy = 7;
            Real[] core = new Real[max_xy * max_xy];

            for (int px = 0; px < max_xy; px++)
            {
                for (int py = 0; py < max_xy; py++)
                {
                    Real r2 = (px - c_xy) * (px - c_xy) + (py - c_xy) * (py - c_xy);
                    core[py * max_xy + px] = Math.Exp(-r2 / sig2) * 1;
                }
            }

            return core;
        }
    }
}
