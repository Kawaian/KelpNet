using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Normalization
{
    //Porting from Chainer finetuning is not implemented yet
    [Serializable]
    public class BatchNormalization : SingleInputFunction
    {
        const string FUNCTION_NAME = "BatchNormalization";

        public bool IsTrain;

        public NdArray Gamma;

        public NdArray Beta;

        public NdArray AvgMean;

        public NdArray AvgVar;


        private readonly Real Decay;
        private readonly Real Eps;

        private Real[] Std;
        private Real[] Xhat;

        private Real[] Mean;
        private Real[] Variance;

        private readonly int ChannelSize;

        public BatchNormalization(int channelSize, double decay = 0.9, double eps = 1e-5, Array initialAvgMean = null, Array initialAvgVar = null, bool isTrain = true, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            ChannelSize = channelSize;
            Decay = decay;
            Eps = eps;
            IsTrain = isTrain;

            Gamma = new NdArray(channelSize);
            Gamma.Data = (RealArray)Enumerable.Repeat((Real)1, channelSize).ToArray();
            Gamma.Name = Name + " Gamma";

            Beta = new NdArray(channelSize);
            Beta.Name = Name + " Beta";

            Parameters = new NdArray[IsTrain ? 2 : 4];

            //Register the parameter to be learned
            Parameters[0] = Gamma;
            Parameters[1] = Beta;

            AvgMean = new NdArray(channelSize);
            AvgMean.Name = Name + " Mean";
            AvgVar = new NdArray(channelSize);
            AvgVar.Name = Name + " Variance";

            if (initialAvgMean != null)
            {
                AvgMean.Data = (RealArray)Real.GetArray(initialAvgMean);
            }

            if (initialAvgVar != null)
            {
                AvgVar.Data = (RealArray)Real.GetArray(initialAvgVar);
            }

            if (!IsTrain)
            {
                Parameters[2] = AvgMean;
                Parameters[3] = AvgVar;
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray ForwardCpu(NdArray x)
        {
            //Acquisition of calculation parameters
            if (IsTrain)
            {
                //Set Mean and Variance of members
                Variance = new Real[ChannelSize];
                for (int i = 0; i < Variance.Length; i++)
                {
                    Variance[i] = 0;
                }

                Mean = new Real[ChannelSize];
                for (int i = 0; i < Mean.Length; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                    {
                        Mean[i] += x.Data[i + index * x.Length];
                    }

                    Mean[i] /= x.BatchCount;
                }

                for (int i = 0; i < Mean.Length; i++)
                {
                    for (int index = 0; index < x.BatchCount; index++)
                    {
                        Variance[i] += (x.Data[i + index * x.Length] - Mean[i]) * (x.Data[i + index * x.Length] - Mean[i]);
                    }

                    Variance[i] /= x.BatchCount;
                }

                for (int i = 0; i < Variance.Length; i++)
                {
                    Variance[i] += Eps;
                }
            }
            else
            {
                Mean = (Real[])AvgMean.Data;
                Variance = (Real[])AvgVar.Data;
            }

            Std = new Real[Variance.Length];
            for (int i = 0; i < Variance.Length; i++)
            {
                Std[i] = Math.Sqrt(Variance[i]);
            }

            //Calculate result
            Xhat = new Real[x.Data.Length];

            Real[] y = new Real[x.Data.Length];

            int dataSize = 1;
            for (int i = 1; i < x.Shape.Length; i++)
            {
                dataSize *= x.Shape[i];
            }

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < ChannelSize; i++)
                {
                    for (int location = 0; location < dataSize; location++)
                    {
                        int index = batchCount * ChannelSize * dataSize + i * dataSize + location;
                        Xhat[index] = (x.Data[index] - Mean[i]) / Std[i];
                        y[index] = Gamma.Data[i] * Xhat[index] + Beta.Data[i];
                    }
                }
            }

            //Update parameters
            if (IsTrain)
            {
                int m = x.BatchCount;
                Real adjust = m / Math.Max(m - 1.0, 1.0); //unbiased estimation

                for (int i = 0; i < AvgMean.Data.Length; i++)
                {
                    AvgMean.Data[i] *= Decay;
                    Mean[i] *= 1 - Decay; //reuse buffer as a temporary
                    AvgMean.Data[i] += Mean[i];

                    AvgVar.Data[i] *= Decay;
                    Variance[i] *= (1 - Decay) * adjust; //reuse buffer as a temporary
                    AvgVar.Data[i] += Variance[i];
                }
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        private void BackwardCpu(NdArray y, NdArray x)
        {
            Beta.ClearGrad();
            Gamma.ClearGrad();

            for (int i = 0; i < ChannelSize; i++)
            {
                for (int j = 0; j < y.BatchCount; j++)
                {
                    Beta.Grad[i] += y.Grad[i + j * y.Length];
                    Gamma.Grad[i] += y.Grad[i + j * y.Length] * Xhat[j * ChannelSize + i];
                }
            }

            if (IsTrain)
            {
                //With learning
                int m = y.BatchCount;

                for (int i = 0; i < ChannelSize; i++)
                {
                    Real gs = Gamma.Data[i] / Std[i];

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        Real val = (Xhat[j * ChannelSize + i] * Gamma.Grad[i] + Beta.Grad[i]) / m;

                        x.Grad[i + j * ChannelSize] += gs * (y.Grad[i + j * y.Length] - val);
                    }
                }
            }
            else
            {
                //No learning
                for (int i = 0; i < ChannelSize; i++)
                {
                    Real gs = Gamma.Data[i] / Std[i];
                    AvgMean.Grad[i] = -gs * Beta.Grad[i];
                    AvgVar.Grad[i] = -0.5 * Gamma.Data[i] / AvgVar.Data[i] * Gamma.Grad[i];

                    for (int j = 0; j < y.BatchCount; j++)
                    {
                        x.Grad[i + j * ChannelSize] += gs * y.Grad[i + j * y.Length];
                    }
                }
            }
        }

        public override NdArray[] Predict(params NdArray[] input)
        {
            NdArray[] result;

            if (IsTrain)
            {
                //Predict does not train
                IsTrain = false;

                result = OnForward(input);

                //Reset Flag
                IsTrain = true;
            }
            else
            {
                result = OnForward(input);
            }

            return result;
        }
    }
}
