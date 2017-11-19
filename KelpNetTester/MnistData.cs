using System;
using KelpNet.Common;
using MNISTLoader;

namespace KelpNetTester
{
    class MnistData
    {
        readonly MnistDataLoader mnistDataLoader = new MnistDataLoader();

        private NdArray[] X;
        private NdArray[] Tx;

        private NdArray[] Y;
        private NdArray[] Ty;

        public MnistData()
        {
            //Training data
            X = new NdArray[mnistDataLoader.TrainData.Length];
            //Training data label
            Tx = new NdArray[mnistDataLoader.TrainData.Length];

            for (int i = 0; i < mnistDataLoader.TrainData.Length; i++)
            {
                Real[] x = new Real[28 * 28];
                for (int j = 0; j < mnistDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = mnistDataLoader.TrainData[i][j] / 255.0;
                }
                X[i] = new NdArray(x, new[] { 1, 28, 28 });

                Tx[i] = new NdArray(new[] { (Real)mnistDataLoader.TrainLabel[i] });
            }

            //Teacher data
            Y = new NdArray[mnistDataLoader.TeachData.Length];
            //Teacher data label
            Ty = new NdArray[mnistDataLoader.TeachData.Length];

            for (int i = 0; i < mnistDataLoader.TeachData.Length; i++)
            {
                Real[] y = new Real[28 * 28];
                for (int j = 0; j < mnistDataLoader.TeachData[i].Length; j++)
                {
                    y[j] = mnistDataLoader.TeachData[i][j] / 255.0;
                }
                Y[i] = new NdArray(y, new[] { 1, 28, 28 });

                Ty[i] = new NdArray(new[] { (Real)mnistDataLoader.TeachLabel[i] });
            }
        }

        public MnistDataSet GetRandomYSet(int dataCount)
        {
            NdArray listY = new NdArray(new[] { 1, 28, 28 }, dataCount);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(Y.Length);

                Array.Copy(Y[index].Data, 0, listY.Data,i * listY.Length,listY.Length);
                listTy.Data[i] = Ty[index].Data[0];
            }

            return new MnistDataSet(listY, listTy);
        }

        public MnistDataSet GetRandomXSet(int dataCount)
        {
            NdArray listX = new NdArray(new[] { 1, 28, 28 }, dataCount);
            NdArray listTx = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(X.Length);

                Array.Copy(X[index].Data, 0, listX.Data, i * listX.Length, listX.Length);
                listTx.Data[i] = Tx[index].Data[0];
            }

            return new MnistDataSet(listX, listTx);
        }
    }

    public class MnistDataSet
    {
        public NdArray Data;
        public NdArray Label;

        public MnistDataSet(NdArray data, NdArray label)
        {
            Data = data;
            Label = label;
        }
    }
}
