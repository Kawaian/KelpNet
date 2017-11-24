using System;
using System.Collections.Generic;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : CompressibleFunction
    {
        const string FUNCTION_NAME = "Linear";

        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "gpuYSum = ForwardActivate(gpuYSum);";

        public NdArray Weight;
        public NdArray Bias;

        public readonly bool NoBias;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames, gpuEnable)
        {
            OutputCount = outputCount;
            InputCount = inputCount;

            Weight = new NdArray(outputCount, inputCount);
            Weight.Name = Name + " Weight";

            NoBias = noBias;

            Parameters = new NdArray[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer.InitWeight(Weight);
            }
            else
            {
                Weight.Data = (RealArray)Real.GetArray(initialW);
            }

            Parameters[0] = Weight;

            if (!noBias)
            {
                Bias = new NdArray(outputCount);
                Bias.Name = Name + " Bias";

                if (initialb != null)
                {
                    Bias.Data = (RealArray)Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }
            
            sw = new Stopwatch();
            sw.Start();
        }

        RealArray GetBiasedValue(int batchCount)
        {
            RealArray y = new RealArray(OutputCount * batchCount);

            for (int i = 0; i < batchCount; i++)
            {
                //realarr
                RealArray.Copy(Bias.Data, 0, y, i * OutputCount, Bias.Data.Length);
            }

            return y;
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray x)
        {
            RealArray y = NoBias ? new RealArray(OutputCount * x.BatchCount) : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    for (int j = 0; j < InputCount; j++)
                    {
                        y[batchCount * OutputCount + i] += x.Data[batchCount * InputCount + j] * Weight.Data[i * InputCount + j];
                    }
                }
            }

            if (Activator != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = Activator.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        protected override void OnGpuEnableChanged()
        {
            Weight.SetGpuEnable(GpuEnable);
            outputY.SetGpuEnable(GpuEnable);
        }

        private Task<ComputeBuffer<T>> CreateBufferAsync<T>(ComputeMemoryFlags flag, T[] data) where T : struct
        {
            var t = Task.Factory.StartNew(()=> { return new ComputeBuffer<T>(Weaver.Context, flag, data); });
            return t;
        }

        // TODO: need to removed
        Stopwatch sw;
        private void ASleep(double i)
        {
            double start = sw.Elapsed.TotalMilliseconds;
            while (true)
            {
                if (sw.Elapsed.TotalMilliseconds - start >= i - 0.9)
                {
                    return;
                }
                Thread.Sleep(1);
            }
        }

        NdArray outputY = null;
        protected override NdArray NeedPreviousForwardGpu(NdArray x)
        {
            var y = NoBias ? new RealArray(OutputCount * x.BatchCount) : GetBiasedValue(x.BatchCount);
            
            var gpuX = x.Data.AsBuffer();
            var gpuW = Weight.Data.AsBuffer();
            NdArray.CopyOrNew(ref outputY, y, GpuEnable);
            var gpuY = outputY.Data.AsBuffer();

            ForwardKernel.SetMemoryArgument(0, gpuX);
            ForwardKernel.SetMemoryArgument(1, gpuW);
            ForwardKernel.SetMemoryArgument(2, gpuY);
            ForwardKernel.SetValueArgument(3, OutputCount);
            ForwardKernel.SetValueArgument(4, InputCount);

            Weaver.CommandQueue.Execute
            (
                ForwardKernel,
                null,
                new long[] { OutputCount, x.BatchCount },
                null,
                null
            );

            Weaver.CommandQueue.Flush();
            ASleep(6.5);
            Weaver.CommandQueue.Finish();

            return NdArray.Convert(outputY, new[] { OutputCount }, x.BatchCount, this);
        }

        RealArray GetActivatedgy(NdArray y)
        {
            RealArray activatedgY = new RealArray(y.Grad.Length);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    int index = batchCount * OutputCount + i;
                    activatedgY[index] = Activator.BackwardActivate(y.Grad[index], y.Data[index]);
                }
            }

            return activatedgY;
        }

        void CalcBiasGrad(RealArray gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    Bias.Grad[i] += gy[batchCounter * OutputCount + i];
                }
            }
        }

        protected override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            RealArray activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    Real gyData = activatedgy[i + batchCount * OutputCount];

                    for (int j = 0; j < InputCount; j++)
                    {
                        Weight.Grad[i * InputCount + j] += x.Data[j + batchCount * InputCount] * gyData;
                        x.Grad[j + batchCount * InputCount] += Weight.Data[i * InputCount + j] * gyData;
                    }
                }
            }
        }
        
        protected override void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            RealArray gx = new RealArray(x.Data.Length);
            gx.ToGpu();
            RealArray activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            activatedgy.ToGpu();
            var gpugY = activatedgy.AsBuffer();

            var gpugW = Weight.Grad.AsBuffer();
            var gpuX = x.Data.AsBuffer();

            BackwardgWKernel.SetMemoryArgument(0, gpugY);
            BackwardgWKernel.SetMemoryArgument(1, gpuX);
            BackwardgWKernel.SetMemoryArgument(2, gpugW);
            BackwardgWKernel.SetValueArgument(3, y.BatchCount);
            BackwardgWKernel.SetValueArgument(4, OutputCount);
            BackwardgWKernel.SetValueArgument(5, InputCount);

            Weaver.CommandQueue.Execute
            (
                BackwardgWKernel,
                null,
                new long[] { InputCount, OutputCount },
                null,
                null
            );

            Weaver.CommandQueue.Finish();
            //Weaver.CommandQueue.ReadFromBuffer(gpugW, ref Weight.Grad, true, null);

            var gpugX = gx.AsBuffer();
            var gpuW = Weight.Data.AsBuffer();

            BackwardgXKernel.SetMemoryArgument(0, gpugY);
            BackwardgXKernel.SetMemoryArgument(1, gpuW);
            BackwardgXKernel.SetMemoryArgument(2, gpugX);
            BackwardgXKernel.SetValueArgument(3, y.BatchCount);
            BackwardgXKernel.SetValueArgument(4, OutputCount);
            BackwardgXKernel.SetValueArgument(5, InputCount);

            Weaver.CommandQueue.Execute
            (
                BackwardgXKernel,
                null,
                new long[] { InputCount, y.BatchCount },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }

            gx.Dispose();
            activatedgy.Dispose();
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
