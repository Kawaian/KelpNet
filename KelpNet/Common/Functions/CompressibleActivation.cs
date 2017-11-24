using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cloo;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class CompressibleActivation : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "Activation";

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel ForwardKernel;

        [NonSerialized]
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public ComputeKernel BackwardKernel;

        //Character string of Activate function for GPU
        public string ActivateFunctionString;

        //Activate virtual function used in.Net
        internal abstract Real ForwardActivate(Real x);
        internal abstract Real BackwardActivate(Real gy, Real y);

        public string ForwardKernelName { get; }
        public string BackwardKernelName { get; }

        protected string ActivateKernelString;

        protected CompressibleActivation(string functionName, KeyValuePair<string, string>[] parameters, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            string kernelNameBase = functionName.Replace(" ", "");
            ForwardKernelName = kernelNameBase + "Forward";
            BackwardKernelName = kernelNameBase + "Backward";

            ActivateKernelString = Weaver.GetKernelSource(FUNCTION_NAME).Replace("/*kernelNameBase*/", kernelNameBase);
            ActivateFunctionString = Weaver.GetKernelSource(functionName);

            if (parameters != null)
            {
                foreach (var parameter in parameters)
                {
                    ActivateFunctionString = ActivateFunctionString.Replace(parameter.Key, parameter.Value);
                }
            }

            SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            GpuEnable = enable & Weaver.Enable;

            if (GpuEnable)
            {
                CreateKernel();
                SingleInputForward = NeedPreviousForwardGpu;
                SingleOutputBackward = NeedPreviousBackwardGpu;
            }
            else
            {
                SingleInputForward = NeedPreviousForwardCpu;
                SingleOutputBackward = NeedPreviousBackwardCpu;
            }

            return GpuEnable;
        }

        public void CreateKernel()
        {
            string kernelSource = ActivateFunctionString + ActivateKernelString;

            ComputeProgram program = Weaver.CreateProgram(kernelSource);
            ForwardKernel = program.CreateKernel(ForwardKernelName);
            BackwardKernel = program.CreateKernel(BackwardKernelName);
        }

        private NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = ForwardActivate(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        protected override void OnGpuEnableChanged()
        {
            if (!GpuEnable)
            {
                if (outputY != null)
                {
                    outputY.Dispose();
                    outputY = null;
                }


            }
        }

        RealArray outputY = null;
        private NdArray NeedPreviousForwardGpu(NdArray x)
        {
            NdArray.CheckLengthAndMayCreate(ref outputY, x.Data.Length, true);

            var gpuX = x.Data.AsBuffer();
            var gpuY = outputY.AsBuffer();

            ForwardKernel.SetMemoryArgument(0, gpuX);
            ForwardKernel.SetMemoryArgument(1, gpuY);

            Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { x.Data.Length },
                    null,
                    null
                );

            Weaver.CommandQueue.Finish();

            return NdArray.Convert(outputY, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += BackwardActivate(y.Grad[i], y.Data[i]);
            }
        }

        RealArray gx = null;
        private void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            NdArray.CheckLengthAndMayCreate(ref gx, y.Grad.Length, true);

            var gpugX = gx.AsBuffer();
            var gpugY = y.Grad.AsBuffer();
            var gpuY = y.Data.AsBuffer();

            BackwardKernel.SetMemoryArgument(0, gpugY);
            BackwardKernel.SetMemoryArgument(1, gpuY);
            BackwardKernel.SetMemoryArgument(2, gpugX);

            Weaver.CommandQueue.Execute
            (
                BackwardKernel,
                null,
                new long[] { y.Grad.Length },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
