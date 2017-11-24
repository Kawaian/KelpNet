using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;
using KelpNet.Functions.Arrays;

namespace KelpNet.Functions.Mathmetrics
{
    [Serializable]
    public class MultiplyScale : SingleInputFunction
    {
        const string FUNCTION_NAME = "MultiplyScale";

        private int Axis;
        public NdArray Weight;
        public NdArray Bias;
        public bool BiasTerm = false;

        public MultiplyScale(int axis = 1, int[] wShape = null, bool biasTerm = false, Array initialW = null, Array initialb = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;
            this.BiasTerm = biasTerm;

#if DEBUG
            if (wShape == null)
            {
                if (biasTerm)
                {
                    throw new Exception("Please use AddBias for use only with Bias");
                }
                else
                {
                    throw new Exception("Parameter setting is incorrect");
                }
            }
#endif
            this.Weight = new NdArray(wShape);
            this.Parameters = new NdArray[biasTerm ? 2 : 1];

            if (initialW == null)
            {
                this.Weight.Data = Enumerable.Repeat((Real) 1.0, Weight.Data.Length).ToArray();
            }
            else
            {
                this.Weight.Data = Real.GetArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (biasTerm)
            {
                this.Bias = new NdArray(wShape);

                if (initialb != null)
                {
                    this.Bias.Data = Real.GetArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray x)
        {
            int[] inputShape = x.Shape;
            int[] outputShape = this.Weight.Shape;

            List<int> shapeList = new List<int>();

            for (int i = 0; i < this.Axis; i++)
            {
                shapeList.Add(1);
            }

            shapeList.AddRange(outputShape);

            for (int i = 0; i < inputShape.Length - this.Axis - outputShape.Length; i++)
            {
                shapeList.Add(1);
            }

            int[] preShape = shapeList.ToArray();

            NdArray y1 = new Reshape(preShape).OnForward(this.Weight)[0];
            NdArray y2 = new Broadcast(inputShape).OnForward(y1)[0];

            if (BiasTerm)
            {
                NdArray b1 = new Reshape(preShape).OnForward(this.Bias)[0];
                NdArray b2 = new Broadcast(inputShape).OnForward(b1)[0];

                return x * y2 + b2;
            }
            else
            {
                return x * y2;
            }
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
        }
    }
}
