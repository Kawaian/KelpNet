using System;

namespace KelpNet.Common.Functions.Type
{
    [Serializable]
    public abstract class DualInputFunction : Function
    {
        protected Func<NdArray, NdArray, NdArray> DualInputForward;
        protected Action<NdArray, NdArray, NdArray> DualOutputBackward;

        protected DualInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;
            xs[1].UseCount++;

            return new[] { DualInputForward(xs[0], xs[1]) };
        }

        public override void Backward(params NdArray[] ys)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 2) throw new Exception("Invalid argument");
#endif
            BackwardCountUp();

            xs[0].UseCount--;
            xs[1].UseCount--;

            DualOutputBackward(ys[0], xs[0], xs[1]);
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return new[] { DualInputForward(xs[0], xs[1]) };
        }
    }
}
