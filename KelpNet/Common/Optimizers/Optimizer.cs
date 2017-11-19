using System;
using System.Collections.Generic;

namespace KelpNet.Common.Optimizers
{
    //Having parameters in the prime class of Optimizer
    [Serializable]
    public abstract class Optimizer
    {
        public long UpdateCount = 1;
        protected List<OptimizerParameter> OptimizerParameters = new List<OptimizerParameter>();

        internal abstract void AddFunctionParameters(NdArray[] functionParameters);

        public void Update()
        {
            bool isUpdated = false;

            for (int i = 0; i < this.OptimizerParameters.Count; i++)
            {
                //Perform discount of slope and check whether there was update
                if (this.OptimizerParameters[i].FunctionParameter.Reduce())
                {
                    this.OptimizerParameters[i].UpdateFunctionParameters();

                    this.OptimizerParameters[i].FunctionParameter.ClearGrad();

                    isUpdated = true;
                }
            }

            if (isUpdated)
            {
                this.UpdateCount++;
            }
        }
    }

    //This class is created with FunctionParameter 1: 1
    [Serializable]
    public abstract class OptimizerParameter
    {
        public NdArray FunctionParameter;

        protected OptimizerParameter(NdArray functionParameter)
        {
            this.FunctionParameter = functionParameter;
        }

        public abstract void UpdateFunctionParameters();
    }
}
