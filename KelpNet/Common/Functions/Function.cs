using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Common.Functions
{
    //Base class of Function stacked in FunctionStack
    [Serializable]
    public abstract class Function
    {
        public string Name;

        private bool _gpuEnable;
        public bool GpuEnable
        {
            get => _gpuEnable;
            protected set
            {
                if (_gpuEnable != value)
                {
                    _gpuEnable = value;
                    OnGpuEnableChanged();
                }
            }
        }

        public NdArray[] Parameters = { };
        public Optimizer[] Optimizers = { };

        public string[] InputNames;
        public string[] OutputNames;

        [NonSerialized]
        public List<NdArray[]> PrevInputs = new List<NdArray[]>();

        //constructor
        protected Function(string name, string[] inputNames = null, string[] outputNames = null)
        {
            Name = name;

            if (inputNames != null)
            {
                InputNames = inputNames.ToArray();
            }

            if (outputNames != null)
            {
                OutputNames = outputNames.ToArray();
            }
        }

        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            PrevInputs = new List<NdArray[]>();
        }

        public NdArray[] Forward(params NdArray[] xs)
        {
            foreach (var item in xs)
            {
                if (GpuEnable)
                {
                    if (!item.IsGpu)
                        item.ToGpu();
                }
                else
                {
                    if (item.IsGpu)
                        item.ToCpu();
                }
            }

            var ret = OnForward(xs);
            return ret;
        }
        protected abstract NdArray[] OnForward(params NdArray[] xs);

        public void Backward(params NdArray[] ys)
        {
            foreach (var item in ys)
            {
                if (GpuEnable)
                {
                    if (!item.IsGpu)
                        item.ToGpu();
                }
                else
                {
                    if (item.IsGpu)
                        item.ToCpu();
                }
            }

            OnBackward(ys);
        }
        protected virtual void OnBackward(params NdArray[] ys) { }

        public virtual void SetOptimizer(params Optimizer[] optimizers)
        {
            Optimizers = optimizers;

            foreach (Optimizer optimizer in optimizers)
            {
                optimizer.AddFunctionParameters(Parameters);
            }
        }

        protected virtual void OnGpuEnableChanged() { }

        //Function to call when updating parameters
        protected void BackwardCountUp()
        {
            foreach (NdArray parameter in Parameters)
            {
                parameter.CountUp();
            }
        }

        //Evaluation function
        public virtual NdArray[] Predict(params NdArray[] input)
        {
            return OnForward(input);
        }

        public virtual void Update()
        {
            foreach (Optimizer optimizer in Optimizers)
            {
                optimizer.Update();
            }
        }

        //A process of returning specific data to the initial value after a certain process is executed
        public virtual void ResetState()
        {
            PrevInputs.Clear();
        }

        //Return name
        public override string ToString()
        {
            return Name;
        }

        //Method to create copy
        public Function Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
