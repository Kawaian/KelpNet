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
                Weight.Data = Real.GetArray(initialW);
            }

            Parameters[0] = Weight;

            if (!noBias)
            {
                Bias = new NdArray(outputCount);
                Bias.Name = Name + " Bias";

                if (initialb != null)
                {
                    Bias.Data = Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }
            
            sw = new Stopwatch();
            sw.Start();
        }

        Real[] GetBiasedValue(int batchCount)
        {
            Real[] y = new Real[OutputCount * batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                Array.Copy(Bias.Data, 0, y, i * OutputCount, Bias.Data.Length);
            }

            return y;
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

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
        
        ComputeBuffer<Real> gpuW;
        protected override void OnGpuEnableChanged()
        {
            if (GpuEnable)
            {
                if(gpuW == null)
                {
                    //copy to gpu memory
                    gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, Weight.Data);
                }
            }
            else
            {
                if(gpuW != null)
                {
                    //copy from gpu memory
                    Weaver.CommandQueue.ReadFromBuffer(gpuW, ref Weight.Data, true, null);
                    gpuW.Dispose();
                    gpuW = null;
                }
            }
        }

        private Task<ComputeBuffer<T>> CreateBufferAsync<T>(ComputeMemoryFlags flag, T[] data) where T : struct
        {
            var t = Task.Factory.StartNew(()=> { return new ComputeBuffer<T>(Weaver.Context, flag, data); });
            return t;
        }

        // TODO: need to removed
        Stopwatch sw;
        private void ASleep(int i)
        {
            double start = sw.Elapsed.TotalMilliseconds;
            while (true)
            {
                if (sw.Elapsed.TotalMilliseconds - start >= i - 0.9)
                    return;
                Thread.Sleep(1);
            }
        }

        protected override NdArray NeedPreviousForwardGpu(NdArray x)
        {
            Real[] y = NoBias ? new Real[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);
            
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> parent of 19034f2... add RealArray; start some opt
            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, y))
            {
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
<<<<<<< HEAD
                //for less cpu use
                ASleep(5);
=======
                //for less cpu use. this is 65% of computation time (10.4ms on 1080ti).
                ASleep(6.5);
>>>>>>> parent of 19034f2... add RealArray; start some opt
                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, new[] { OutputCount }, x.BatchCount, this);
<<<<<<< HEAD
>>>>>>> parent of 8f322f3... hmm
=======
>>>>>>> parent of 19034f2... add RealArray; start some opt
        }

        Real[] GetActivatedgy(NdArray y)
        {
            Real[] activatedgY = new Real[y.Grad.Length];

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

        void CalcBiasGrad(Real[] gy, int batchCount)
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
            Real[] activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
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
            Real[] gx = new Real[x.Data.Length];
            Real[] activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, activatedgy))
            {
                using (ComputeBuffer<Real> gpugW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, Weight.Grad))
                using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
                {
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
                    Weaver.CommandQueue.ReadFromBuffer(gpugW, ref Weight.Grad, true, null);
                }

                using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
                using (ComputeBuffer<Real> gpuW = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, Weight.Data))
                {
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
                    Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }

        public Convolution2D AsConvolution2D()
        {
            return new Convolution2D(this);
        }
    }
}
