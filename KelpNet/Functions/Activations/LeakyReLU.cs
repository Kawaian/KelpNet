﻿using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousOutputFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU", bool isParallel = true) : base(name,isParallel)
        {
            this._slope = slope;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = x.Data[i] < 0 ? x.Data[i] *= this._slope : x.Data[i];
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : prevOutput.Data[i] * this._slope;
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
