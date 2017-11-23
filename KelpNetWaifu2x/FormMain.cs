using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using System.Diagnostics;
using System.Dynamic;

namespace KelpNetWaifu2x
{
    /* Please download the model file from https://github.com/nagadomi/waifu2x/tree/master/models/upconv_7/art */
    /* The sample is confirmed to operate on scale 2.0 x _ model json */

    public partial class FormMain : Form
    {
        FunctionStack nn;

        public FormMain()
        {
            InitializeComponent();

            //Initialize GPU
            Weaver.Initialize(ComputeDeviceTypes.Cpu);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "Json file (*.Json) | *.json; | all files (*.*) | *.*",
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                int layerCounter = 1;

                var json = DynamicJson.Parse(File.ReadAllText(ofd.FileName));

                List<Function> functionList = new List<Function>();

                //Please ignore Microsoft.CSharp.RuntimeBinder.RuntimeBinderException
                foreach (var data in json)
                {
                    Real[,,,] weightData = new Real[(int)data["nOutputPlane"], (int)data["nInputPlane"], (int)data["kW"], (int)data["kH"]];
                    var target = (double[][][][])data["weight"];
                    Console.WriteLine($"HEYYYYYYYYYYYYYYYYYY:L =-========    {weightData.GetLength(0)},{weightData.GetLength(1)},{weightData.GetLength(2)},{weightData.GetLength(3)}");
                    Console.WriteLine($"TTTTTTTTTTTTTTTTTTTT:L =-========    {target.GetLength(0)},{target[0].GetLength(0)},{target[0][0].GetLength(0)},{target[0][0][0].GetLength(0)}");
                    Console.WriteLine($"NANIIIIIIIIIIIIIIIDD:L =-========    ");

                    for (int i = 0; i < weightData.GetLength(0); i++)
                    {
                        for (int j = 0; j < weightData.GetLength(1); j++)
                        {
                            for (int k = 0; k < weightData.GetLength(2); k++)
                            {
                                for (int l = 0; l < weightData.GetLength(3); l++)
                                {
                                    // test 00 1 0
                                    // test 01 1 0.2
                                    // test 10 1 0.35
                                    // test 11 1 0.35
                                    // ct 00
                                    // ct 01
                                    // ct 10
                                    // ct 11 1 
                                    //idk some times it is weird
                                    if (weightData.GetLength(0) == target.GetLength(0))
                                        weightData[i, j, k, l] = target[i][j][k][l];
                                    else
                                        weightData[i, j, k, l] = target[j][i][k][l];
                                    //target.TryGetIndex(null, new object[] { i }, out object d);
                                    //((DynamicJson)d).TryGetIndex(null, new object[] { j }, out object dd);
                                    //((DynamicJson)dd).TryGetIndex(null, new object[] { k }, out object ddd);
                                    //((DynamicJson)ddd).TryGetIndex(null, new object[] { l }, out object dddd);
                                    //weightData[i, j, k, l] = (double)dddd;
                                }
                            }
                        }
                    }

                    //Make a pad and adjust the size of the input and the output image
                    functionList.Add(new Convolution2D((int)data["nInputPlane"], (int)data["nOutputPlane"], (int)data["kW"], pad: (int)data["kW"] / 2, initialW: weightData, initialb: (Real[])data["bias"],name: "Convolution2D l" + layerCounter++, gpuEnable: false));
                    functionList.Add(new LeakyReLU(0.1, name: "LeakyReLU l" + layerCounter++));
                }

                nn = new FunctionStack(functionList.ToArray());
                nn.Compress();

                MessageBox.Show("Read complete");
            }
        }

        Bitmap _baseImage;
        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "Image file (*. Jpg; *. Png) | *.Jpg; *.png; | All files (*. *) | *. *"
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                this._baseImage = new Bitmap(ofd.FileName);
                this.pictureBox1.Image = new Bitmap(this._baseImage);
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog
            {
                Filter = "png file (*.png) | *.png | all files (*. *) | *. *",
                FileName = "result.png"
            };

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                Task.Factory.StartNew(() =>
                {
                    //It is necessary to enlarge in advance before entering the network
                    Bitmap resultImage = new Bitmap(this._baseImage.Width * 2, this._baseImage.Height * 2, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);

                    //Use nearest neighbor for interpolation
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;

                    //Draw an image enlarged
                    g.DrawImage(this._baseImage, 0, 0, this._baseImage.Width * 2, this._baseImage.Height * 2);
                    g.Dispose();

                    NdArray image = NdArrayConverter.Image2NdArray(resultImage);
                    NdArray[] resultArray = this.nn.Predict(image);
                    resultImage = NdArrayConverter.NdArray2Image(resultArray[0].GetSingleArray(0));
                    resultImage.Save(sfd.FileName);
                    this.pictureBox1.Image = new Bitmap(resultImage);
                }
                    ).ContinueWith(_ =>
                    {
                        MessageBox.Show("Conversion complete");
                    });

                MessageBox.Show("Conversion processing has been started. \n Please wait for a while until \"Conversion completed\" message is displayed \n * It will take a very long time (about three minutes with 64 x 64 images)");
            }
        }
    }
}
