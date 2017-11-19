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
            this.ForwardKernelName = kernelNameBase + "Forward";
            this.BackwardKernelName = kernelNameBase + "Backward";

            this.ActivateKernelString = Weaver.GetKernelSource(FUNCTION_NAME).Replace("/*kernelNameBase*/", kernelNameBase);
            this.ActivateFunctionString = Weaver.GetKernelSource(functionName);

            if (parameters != null)
            {
                foreach (var parameter in parameters)
                {
                    this.ActivateFunctionString = this.ActivateFunctionString.Replace(parameter.Key, parameter.Value);
                }
            }

            this.SetGpuEnable(gpuEnable);
        }

        public bool SetGpuEnable(bool enable)
        {
            this.GpuEnable = enable & Weaver.Enable;

            if (this.GpuEnable)
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
            string kernelSource = this.ActivateFunctionString + this.ActivateKernelString;

            ComputeProgram program = Weaver.CreateProgram(kernelSource);
            this.ForwardKernel = program.CreateKernel(this.ForwardKernelName);
            this.BackwardKernel = program.CreateKernel(this.BackwardKernelName);
        }

        private NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = this.ForwardActivate(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        private NdArray NeedPreviousForwardGpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            using (ComputeBuffer<Real> gpuX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, x.Data))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, y.Length))
            {
                this.ForwardKernel.SetMemoryArgument(0, gpuX);
                this.ForwardKernel.SetMemoryArgument(1, gpuY);

                Weaver.CommandQueue.Execute
                    (
                        this.ForwardKernel,
                        null,
                        new long[] { x.Data.Length },
                        null,
                        null
                    );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref y, true, null);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += this.BackwardActivate(y.Grad[i], y.Data[i]);
            }
        }

        private void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            Real[] gx = new Real[y.Grad.Length];

            using (ComputeBuffer<Real> gpugY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, y.Grad))
            using (ComputeBuffer<Real> gpuY = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, y.Data))
            using (ComputeBuffer<Real> gpugX = new ComputeBuffer<Real>(Weaver.Context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.AllocateHostPointer, gx.Length))
            {
                this.BackwardKernel.SetMemoryArgument(0, gpugY);
                this.BackwardKernel.SetMemoryArgument(1, gpuY);
                this.BackwardKernel.SetMemoryArgument(2, gpugX);

                Weaver.CommandQueue.Execute
                    (
                        this.BackwardKernel,
                        null,
                        new long[] { y.Grad.Length },
                        null,
                        null
                    );

                Weaver.CommandQueue.Finish();
                Weaver.CommandQueue.ReadFromBuffer(gpugX, ref gx, true, null);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
