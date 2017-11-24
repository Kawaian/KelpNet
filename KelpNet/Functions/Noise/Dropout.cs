using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "Dropout";

        private readonly Real dropoutRatio;
        private readonly List<Real[]> maskStack = new List<Real[]>();

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.dropoutRatio = dropoutRatio;

            SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            GpuEnable = enable & Weaver.Enable;

            if (GpuEnable)
            {
                CreateKernel();
                SingleInputForward = ForwardGpu;
                SingleOutputBackward = BackwardGpu;
            }
            else
            {
                SingleInputForward = ForwardCpu;
                SingleOutputBackward = BackwardCpu;
            }

            return GpuEnable;
        }

        public void CreateKernel()
        {
            string kernelSource = Weaver.GetKernelSource(FUNCTION_NAME);
            ComputeProgram program = Weaver.CreateProgram(kernelSource);

            ForwardKernel = program.CreateKernel("DropoutForward");
            BackwardKernel = program.CreateKernel("DropoutBackward");
        }

        private Real[] MakeMask(int xLength)
        {
            Real[] mask = new Real[xLength];
            Real scale = 1 / (1 - dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= dropoutRatio ? scale : 0;
            }

            maskStack.Add(mask);

            return mask;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        protected override void OnGpuEnableChanged()
        {
            if (!GpuEnable)
            {
                if (gpuResult != null)
                {
                    gpuResult.Dispose();
                    gpuResult = null;
                }

                if(gpuMask != null)
                {
                    gpuMask.Dispose();
                    gpuMask = null;
                }
            }
        }

        RealArray gpuResult = null;
        RealArray gpuMask = null;
        public NdArray ForwardGpu(NdArray x)
        {
            Real[] mask = MakeMask(x.Length);

            var gpuX = x.Data.AsBuffer();
            NdArray.CheckLengthAndMayCreate(ref gpuResult, x.Data.Length);
            var gpuY = gpuResult.AsBuffer();
            NdArray.CheckLengthAndMayCreate(ref gpuMask, x.Length);
            RealArray.Copy((RealArray)mask, gpuMask);
            var gpuMaskBuf = gpuMask.AsBuffer();

            ForwardKernel.SetMemoryArgument(0, gpuX);
            ForwardKernel.SetMemoryArgument(1, gpuMaskBuf);
            ForwardKernel.SetMemoryArgument(2, gpuY);
            ForwardKernel.SetValueArgument(3, mask.Length);

            Weaver.CommandQueue.Execute
            (
                ForwardKernel,
                null,
                new long[] { x.Data.Length },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            return NdArray.Convert(gpuResult, x.Shape, x.BatchCount, this);
        }

        public void BackwardCpu(NdArray y, NdArray x)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = maskStack[maskStack.Count - 1];
            maskStack.RemoveAt(maskStack.Count - 1);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[b * y.Length + i] *= mask[i];
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }

        public void BackwardGpu(NdArray y, NdArray x)
        {
            Real[] mask = maskStack[maskStack.Count - 1];
            maskStack.RemoveAt(maskStack.Count - 1);

            NdArray.CheckLengthAndMayCreate(ref gpuMask, x.Length);
            RealArray.Copy((RealArray)mask, gpuMask);
            var gpuMaskBuf = gpuMask.AsBuffer();
            var gpugX = y.Grad.AsBuffer();

            BackwardKernel.SetMemoryArgument(0, gpuMaskBuf);
            BackwardKernel.SetMemoryArgument(1, gpugX);
            BackwardKernel.SetValueArgument(2, y.Length);

            Weaver.CommandQueue.Execute
            (
                BackwardKernel,
                null,
                new long[] { mask.Length, y.BatchCount },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i];
            }
        }


        //I do not do anything when Predict
        public override NdArray Predict(NdArray input)
        {
            return input;
        }
    }
}
