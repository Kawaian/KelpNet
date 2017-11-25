using System;
using System.Collections.Generic;
using KelpNet.Common.Optimizers;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;

namespace KelpNet.Common.Functions.Container
{
    //The main class of this library for stacking layers.
    //A set of functions that are executed simultaneously in one Forward, Backward, Update.
    [Serializable]
    public class FunctionStack : Function
    {
        const string FUNCTION_NAME = "FunctionStack";

        public static void SwitchToGPU(FunctionStack functionStack)
        {
            foreach (Function function in functionStack.Functions)
            {
                if (function is IParallelizable)
                {
                    ((IParallelizable)function).SetGpuEnable(true);
                }

                if (function is SplitFunction)
                {
                    SplitFunction splitFunction = (SplitFunction)function;
                    for (int i = 0; i < splitFunction.SplitedFunctions.Length; i++)
                    {
                        SwitchToGPU(splitFunction.SplitedFunctions[i]);
                    }
                }
            }

            //Perform layer compression on a block-by-block basis
            functionStack.Compress();
        }

        //All layers are stored here
        public Function[] Functions { get; private set; }

        public FunctionStack(Function[] functions, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = functions;
        }

        public FunctionStack(Function function, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = new[] { function };
        }

        public FunctionStack(params Function[] functions) : base(FUNCTION_NAME)
        {
            this.Functions = new Function[]{};
            this.Add(functions);
        }

        //Since it is an inefficient implementation
        public void Add(params Function[] function)
        {
            if (function != null && function.Length > 0)
            {
                List<Function> functionList = new List<Function>();

                if (this.Functions != null)
                {
                    functionList.AddRange(this.Functions);
                }

                for (int i = 0; i < function.Length; i++)
                {
                    if (function[i] != null) functionList.Add(function[i]);
                }

                this.Functions = functionList.ToArray();

                InputNames = Functions[0].InputNames;
                OutputNames = Functions[Functions.Length - 1].OutputNames;
            }
        }

        public void Compress()
        {
            List<Function> functionList = new List<Function>(Functions);

            //Compress layer
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is CompressibleFunction)
                {
                    if (functionList[i + 1] is CompressibleActivation)
                    {
                        ((CompressibleFunction)functionList[i]).SetActivation((CompressibleActivation)functionList[i + 1]);
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            this.Functions = functionList.ToArray();
        }

        //Forward
        protected override NdArray[] OnForward(params NdArray[] xs)
        {
            NdArray[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Forward(ys);
            }

            return ys;
        }

        //Backward
        protected override void OnBackward(params NdArray[] ys)
        {
            NdArray.Backward(ys[0]);
        }

        //Weight update process
        public override void Update()
        {
            foreach (var function in Functions)
            {
                function.Update();
            }
        }

        //A process of returning specific data to the initial value after a certain process is executed
        public override void ResetState()
        {
            foreach (Function function in this.Functions)
            {
                function.ResetState();
            }
        }

        //Execute forecast
        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Predict(ys);
            }

            return ys;
        }

        public override void SetOptimizer(params Optimizer[] optimizers)
        {
            foreach (Function function in this.Functions)
            {
                function.SetOptimizer(optimizers);
            }
        }
    }
}
