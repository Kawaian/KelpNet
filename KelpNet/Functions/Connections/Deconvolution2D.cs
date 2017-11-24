using System;
using System.Collections.Generic;
using System.Drawing;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Deconvolution2D : CompressibleFunction
    {
        const string FUNCTION_NAME = "Deconvolution2D";
        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "result = ForwardActivate(result);";

        public NdArray Weight;
        public NdArray Bias;

        public readonly bool NoBias;

        private readonly int _kWidth;
        private readonly int _kHeight;
        private readonly int _subSampleX;
        private readonly int _subSampleY;
        private readonly int _trimX;
        private readonly int _trimY;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Deconvolution2D(int inputChannels, int outputChannels, int kSize, int subSample = 1, int trim = 0, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, new []{new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE)}, name, inputNames, outputNames, gpuEnable)
        {
            _kWidth = kSize;
            _kHeight = kSize;
            _trimX = trim;
            _trimY = trim;
            _subSampleX = subSample;
            _subSampleY = subSample;
            NoBias = noBias;

            Parameters = new NdArray[noBias ? 1 : 2];

            OutputCount = outputChannels;
            InputCount = inputChannels;

            Initialize(initialW, initialb);
        }

        public Deconvolution2D(int inputChannels, int outputChannels, Size kSize, Size subSample = new Size(), Size trim = new Size(), bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(FUNCTION_NAME, activation, new []{new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE)}, name, inputNames, outputNames, gpuEnable)
        {
            if (subSample == Size.Empty)
                subSample = new Size(1, 1);

            if (trim == Size.Empty)
                trim = new Size(0, 0);

            _kWidth = kSize.Width;
            _kHeight = kSize.Height;
            _trimX = trim.Width;
            _trimY = trim.Height;
            NoBias = noBias;

            _subSampleX = subSample.Width;
            _subSampleY = subSample.Height;

            Parameters = new NdArray[noBias ? 1 : 2];

            OutputCount = outputChannels;
            InputCount = inputChannels;

            Initialize(initialW, initialb);
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            Weight = new NdArray(OutputCount, InputCount, _kHeight, _kWidth);
            Weight.Name = Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(Weight);
            }
            else
            {
                Weight.Data = (RealArray)Real.GetArray(initialW);
            }

            Parameters[0] = Weight;


            if (!NoBias)
            {
                Bias = new NdArray(OutputCount);
                Bias.Name = Name + " Bias";

                if (initialb != null)
                {
                    Bias.Data = (RealArray)Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }
        }

        protected override NdArray NeedPreviousForwardCpu(NdArray input)
        {
            int outputHeight = (input.Shape[1] - 1) * _subSampleY + _kHeight - _trimY * 2;
            int outputWidth = (input.Shape[2] - 1) * _subSampleX + _kWidth - _trimX * 2;

            Real[] result = new Real[input.BatchCount * OutputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;
            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = Weight.Shape[2] * Weight.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < OutputCount; och++)
                {
                    for (int oy = _trimY; oy < outputHeight + _trimY; oy++)
                    {
                        int iyLimit = oy / _subSampleY + 1 < input.Shape[1] ? oy / _subSampleY + 1 : input.Shape[1];
                        int iyStart = oy - Weight.Shape[2] < 0 ? 0 : (oy - Weight.Shape[2]) / _subSampleY + 1;

                        for (int ox = _trimX; ox < outputWidth + _trimX; ox++)
                        {
                            int ixLimit = ox / _subSampleX + 1 < input.Shape[2] ? ox / _subSampleX + 1 : input.Shape[2];
                            int ixStart = ox - Weight.Shape[3] < 0 ? 0 : (ox - Weight.Shape[3]) / _subSampleX + 1;

                            int outputIndex = batchCount * OutputCount * outSizeOffset + och * outSizeOffset + (oy - _trimY) * outputWidth + ox - _trimX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;
                                int kernelIndexOffset = och * Weight.Shape[1] * kSizeOffset + ich * kSizeOffset;

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                        int kernelIndex = kernelIndexOffset + (oy - iy * _subSampleY) * Weight.Shape[3] + (ox - ix * _subSampleX);

                                        result[outputIndex] += input.Data[inputIndex] * Weight.Data[kernelIndex];
                                    }
                                }
                            }

                        }
                    }
                }
            }

            if (Activator != null && !NoBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < OutputCount; och++)
                    {
                        for (int oy = _trimY; oy < outputHeight + _trimY; oy++)
                        {
                            for (int ox = _trimX; ox < outputWidth + _trimX; ox++)
                            {
                                int outputIndex = batchCount * OutputCount * outSizeOffset + och * outSizeOffset + (oy - _trimY) * outputWidth + ox - _trimX;

                                result[outputIndex] += Bias.Data[och];
                                result[outputIndex] = Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < OutputCount; och++)
                    {
                        for (int oy = _trimY; oy < outputHeight + _trimY; oy++)
                        {
                            for (int ox = _trimX; ox < outputWidth + _trimX; ox++)
                            {
                                int outputIndex = batchCount * OutputCount * outSizeOffset + och * outSizeOffset + (oy - _trimY) * outputWidth + ox - _trimX;

                                result[outputIndex] += Bias.Data[och];
                            }
                        }
                    }
                }
            }
            else if (Activator != null)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < OutputCount; och++)
                    {
                        for (int oy = _trimY; oy < outputHeight + _trimY; oy++)
                        {
                            for (int ox = _trimX; ox < outputWidth + _trimX; ox++)
                            {
                                int outputIndex = batchCount * OutputCount * outSizeOffset + och * outSizeOffset + (oy - _trimY) * outputWidth + ox - _trimX;

                                result[outputIndex] = Activator.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        protected override void OnGpuEnableChanged()
        {
            if (!GpuEnable)
            {
                if(gpuResult != null)
                {
                    gpuResult.Dispose();
                    gpuResult = null;
                }
                if(gpuBias != null)
                {
                    gpuBias.Dispose();
                    gpuBias = null;
                }
            }

            Weight.SetGpuEnable(GpuEnable);
        }

        RealArray gpuResult = null;
        RealArray gpuBias = null;
        protected override NdArray NeedPreviousForwardGpu(NdArray input)
        {
            int outputHeight = (input.Shape[1] - 1) * _subSampleY + _kHeight - _trimY * 2;
            int outputWidth = (input.Shape[2] - 1) * _subSampleX + _kWidth - _trimX * 2;
            
            NdArray.CheckLengthAndMayCreate(ref gpuResult, input.BatchCount * OutputCount * outputWidth * outputHeight, GpuEnable);
            var gpuY = gpuResult.AsBuffer();
            var gpuX = input.Data.AsBuffer();
            var gpuW = Weight.Data.AsBuffer();
            var bias = NoBias ? new RealArray(OutputCount) : Bias.Data;
            NdArray.CopyOrNew(ref gpuBias, bias, GpuEnable);
            var gpub = gpuBias.AsBuffer();
            
            ForwardKernel.SetMemoryArgument(0, gpuX);
            ForwardKernel.SetMemoryArgument(1, gpuW);
            ForwardKernel.SetMemoryArgument(2, gpub);
            ForwardKernel.SetMemoryArgument(3, gpuY);
            ForwardKernel.SetValueArgument(4, input.Shape[1]);
            ForwardKernel.SetValueArgument(5, input.Shape[2]);
            ForwardKernel.SetValueArgument(6, input.Length);
            ForwardKernel.SetValueArgument(7, outputWidth);
            ForwardKernel.SetValueArgument(8, outputHeight);
            ForwardKernel.SetValueArgument(9, _subSampleX);
            ForwardKernel.SetValueArgument(10, _subSampleY);
            ForwardKernel.SetValueArgument(11, _trimX);
            ForwardKernel.SetValueArgument(12, _trimY);
            ForwardKernel.SetValueArgument(13, _kHeight);
            ForwardKernel.SetValueArgument(14, _kWidth);
            ForwardKernel.SetValueArgument(15, OutputCount);
            ForwardKernel.SetValueArgument(16, InputCount);

            Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount * OutputCount, outputHeight, outputWidth },
                    null,
                    null
                );

            Weaver.CommandQueue.Finish();

            return NdArray.Convert(gpuResult, new[] { OutputCount, outputHeight, outputWidth }, input.BatchCount, this);
        }

        Real[] GetActivatedgy(NdArray y)
        {
            int gyIndex = 0;

            Real[] activatedgy = new Real[y.Grad.Length];

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                for (int och = 0; och < y.Shape[0]; och++)
                {
                    for (int olocation = 0; olocation < y.Shape[1] * y.Shape[2]; olocation++)
                    {
                        activatedgy[gyIndex] = Activator.BackwardActivate(y.Grad[gyIndex], y.Data[gyIndex]);
                        gyIndex++;
                    }
                }
            }

            return activatedgy;
        }

        void CalcBiasGrad(Real[] gy, int[] gyShape, int batchCount)
        {
            int gyIndex = 0;

            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int och = 0; och < gyShape[0]; och++)
                {
                    for (int olocation = 0; olocation < gyShape[1] * gyShape[2]; olocation++)
                    {
                        Bias.Grad[och] += gy[gyIndex];

                        gyIndex++;
                    }
                }
            }
        }

        protected override void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            //Real [] gx = new Real [x.Data.Length];
            Real[] activatedgy = Activator != null ? GetActivatedgy(y) : y.Grad.AsArray();
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            //Original logic
            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int och = 0; och < OutputCount; och++)
                {
                    int outChOffset = och * Weight.Shape[1] * Weight.Shape[2] * Weight.Shape[3];
                    int inputOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = _trimY; oy < y.Shape[1] + _trimY; oy++)
                    {
                        int iyLimit = oy / _subSampleY + 1 < x.Shape[1] ? oy / _subSampleY + 1 : x.Shape[1];
                        int iyStart = oy - Weight.Shape[2] < 0 ? 0 : (oy - Weight.Shape[2]) / _subSampleY + 1;

                        for (int ox = _trimX; ox < y.Shape[2] + _trimX; ox++)
                        {
                            int ixLimit = ox / _subSampleX + 1 < x.Shape[2] ? ox / _subSampleX + 1 : x.Shape[2];
                            int ixStart = ox - Weight.Shape[3] < 0 ? 0 : (ox - Weight.Shape[3]) / _subSampleX + 1;

                            int gyIndex = batchCount * y.Length + inputOffset + (oy - _trimY) * y.Shape[2] + ox - _trimX;
                            Real gyData = activatedgy[gyIndex];

                            for (int ich = 0; ich < InputCount; ich++)
                            {
                                int inChOffset = outChOffset + ich * Weight.Shape[2] * Weight.Shape[3];
                                int pinputOffset = batchCount * x.Length + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int pInIndex = pinputOffset + iy * x.Shape[2] + ix;
                                        int gwIndex = inChOffset + (oy - iy * _subSampleY) * Weight.Shape[3] + (ox - ix * _subSampleX);

                                        Weight.Grad[gwIndex] += x.Data[pInIndex] * gyData;
                                        x.Grad[pInIndex] += Weight.Data[gwIndex] * gyData;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        RealArray gpuGx = null;
        protected override void NeedPreviousBackwardGpu(NdArray y, NdArray x)
        {
            NdArray.CheckLengthAndMayCreate(ref gpuGx, x.Data.Length, GpuEnable);
            RealArray activatedgy = Activator != null ? (RealArray)GetActivatedgy(y) : y.Grad;
            activatedgy.ToCpu();
            if (!NoBias) CalcBiasGrad(activatedgy.AsArray(), y.Shape, y.BatchCount);
            activatedgy.ToGpu();

            var gpugY = activatedgy.AsBuffer();
            var gpugW = Weight.Grad.AsBuffer();
            var gpuX = x.Data.AsBuffer();

            BackwardgWKernel.SetMemoryArgument(0, gpugY);
            BackwardgWKernel.SetMemoryArgument(1, gpuX);
            BackwardgWKernel.SetMemoryArgument(2, gpugW);
            BackwardgWKernel.SetValueArgument(3, y.BatchCount);
            BackwardgWKernel.SetValueArgument(4, InputCount);
            BackwardgWKernel.SetValueArgument(5, y.Length);
            BackwardgWKernel.SetValueArgument(6, y.Shape[1]);
            BackwardgWKernel.SetValueArgument(7, y.Shape[2]);
            BackwardgWKernel.SetValueArgument(8, x.Shape[1]);
            BackwardgWKernel.SetValueArgument(9, x.Shape[2]);
            BackwardgWKernel.SetValueArgument(10, x.Length);
            BackwardgWKernel.SetValueArgument(11, _subSampleX);
            BackwardgWKernel.SetValueArgument(12, _subSampleY);
            BackwardgWKernel.SetValueArgument(13, _trimX);
            BackwardgWKernel.SetValueArgument(14, _trimY);
            BackwardgWKernel.SetValueArgument(15, _kHeight);
            BackwardgWKernel.SetValueArgument(16, _kWidth);

            Weaver.CommandQueue.Execute
            (
                BackwardgWKernel,
                null,
                new long[] { OutputCount * InputCount, _kHeight, _kWidth },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            var gpugX = gpuGx.AsBuffer();
            var gpuW = gpuGx.AsBuffer();

            BackwardgXKernel.SetMemoryArgument(0, gpugY);
            BackwardgXKernel.SetMemoryArgument(1, gpuW);
            BackwardgXKernel.SetMemoryArgument(2, gpugX);
            BackwardgXKernel.SetValueArgument(3, OutputCount);
            BackwardgXKernel.SetValueArgument(4, InputCount);
            BackwardgXKernel.SetValueArgument(5, y.Length);
            BackwardgXKernel.SetValueArgument(6, y.Shape[1]);
            BackwardgXKernel.SetValueArgument(7, y.Shape[2]);
            BackwardgXKernel.SetValueArgument(8, x.Shape[1]);
            BackwardgXKernel.SetValueArgument(9, x.Shape[2]);
            BackwardgXKernel.SetValueArgument(10, x.Length);
            BackwardgXKernel.SetValueArgument(11, _subSampleX);
            BackwardgXKernel.SetValueArgument(12, _subSampleY);
            BackwardgXKernel.SetValueArgument(13, _trimX);
            BackwardgXKernel.SetValueArgument(14, _trimY);
            BackwardgXKernel.SetValueArgument(15, _kHeight);
            BackwardgXKernel.SetValueArgument(16, _kWidth);

            Weaver.CommandQueue.Execute
            (
                BackwardgXKernel,
                null,
                new long[] { y.BatchCount * x.Shape[0], x.Shape[1], x.Shape[2] },
                null,
                null
            );

            Weaver.CommandQueue.Finish();

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gpuGx[i];
            }

            if (Activator != null)
                activatedgy.Dispose();
        }
    }
}
