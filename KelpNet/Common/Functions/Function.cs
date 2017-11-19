using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Common.Functions
{
    //Base class of Function stacked in FunctionStack
    [Serializable]
    public abstract class Function
    {
        public string Name;

        public bool GpuEnable { get; protected set; }

        public NdArray[] Parameters = { };
        public Optimizer[] Optimizers = { };

        [NonSerialized]
        public List<NdArray[]> PrevInputs = new List<NdArray[]>();

        public abstract NdArray[] Forward(params NdArray[] xs);
        public virtual void Backward(params NdArray[] ys){}

        public string[] InputNames;
        public string[] OutputNames;

        //constructor
        protected Function(string name, string[] inputNames = null, string[] outputNames = null)
        {
            this.Name = name;

            if (inputNames != null)
            {
                this.InputNames = inputNames.ToArray();
            }

            if (outputNames != null)
            {
                this.OutputNames = outputNames.ToArray();
            }
        }

        public virtual void SetOptimizer(params Optimizer[] optimizers)
        {
            this.Optimizers = optimizers;

            foreach (Optimizer optimizer in optimizers)
            {
                optimizer.AddFunctionParameters(this.Parameters);
            }
        }

        //Function to call when updating parameters
        protected void BackwardCountUp()
        {
            foreach (NdArray parameter in this.Parameters)
            {
                parameter.CountUp();
            }
        }

        //Evaluation function
        public virtual NdArray[] Predict(params NdArray[] input)
        {
            return this.Forward(input);
        }

        public virtual void Update()
        {
            foreach (Optimizer optimizer in this.Optimizers)
            {
                optimizer.Update();
            }
        }

        //A process of returning specific data to the initial value after a certain process is executed
        public virtual void ResetState()
        {
        }

        //Return name
        public override string ToString()
        {
            return this.Name;
        }

        //Method to create copy
        public Function Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
