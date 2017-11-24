using System;
using System.Drawing;
using System.Windows.Forms;
using KelpNet.Common;
using KelpNet.Common.Tools;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    public partial class Test13WinForm : Form
    {
        Deconvolution2D model;
        private Deconvolution2D decon_core;
        private SGD optimizer;
        MeanSquaredError meanSquaredError = new MeanSquaredError();
        private int counter = 0;

        public Test13WinForm()
        {
            this.InitializeComponent();

            ClientSize = new Size(128 * 4, 128 * 4);

            //Create a target filter (If it is practical, here is an unknown value)
            this.decon_core = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true)
            {
                Weight = { Data = (RealArray)MakeOneCore() }
            };

            this.model = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true);

            this.optimizer = new SGD(learningRate: 0.01); //When it is big, it diverges.
            this.model.SetOptimizer(this.optimizer);
        }

        static NdArray getRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            //Randomly make 0.1% points
            Real[] img_p = new Real[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother.Dice.Next(0, 10000);
                img_p[i] = img_p[i] < 10 ? 255 : 0;
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

        private void timer1_Tick(object sender, EventArgs e)
        {
            //At the transplant source, we are educating with the same educational image, but changing to learning closer to practice
            if (this.counter < 11)
            {
                //Generate random dotted images
                NdArray img_p = getRandomImage();

                //Output a learning image with a target filter
                NdArray[] img_core = this.decon_core.Forward(img_p);

                //Output an image with an unlearned filter
                NdArray[] img_y = this.model.Forward(img_p);

                //Implicitly use img_y as NdArray
                this.BackgroundImage = NdArrayConverter.NdArray2Image(img_y[0].GetSingleArray(0));

                Real loss = this.meanSquaredError.Evaluate(img_y, img_core);

                this.model.Backward(img_y);
                this.model.Update();

                this.Text = "[epoch" + this.counter + "] Loss : " + string.Format("{0:F4}", loss);

                this.counter++;
            }
            else
            {
                this.timer1.Enabled = false;
            }

        }
    }
}
