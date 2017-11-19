using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common.Functions.Type;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions.Container
{
    [Serializable]
    public class FunctionDictionary : Function
    {
        const string FUNCTION_NAME = "FunctionDictionary";

        //Managed with Function Record, input / output key added added the function.
        public Dictionary<string, FunctionStack> FunctionBlockDictionary = new Dictionary<string, FunctionStack>();

        //Dictionary holding names of division functions
        public Dictionary<string, FunctionStack> SplitedFunctionDictionary = new Dictionary<string, FunctionStack>();

        //Dictionary execution order list
        public List<FunctionStack> FunctionBlocks = new List<FunctionStack>();

        private readonly bool _compress = false;

        public FunctionDictionary(bool compress = false, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._compress = compress;
        }

        public void Add(Function function)
        {
            if (_compress && //Summarize each branch or not
                (function is SingleInputFunction || function is MultiOutputFunction)) //Gather only on function
            {
                //Check if it was registered in the dict
                if (this.FunctionBlockDictionary.ContainsKey(function.InputNames[0]))
                {
                    //If the block is already registered, it is concatenated to the block
                    this.FunctionBlockDictionary[function.InputNames[0]].Add(function);

                    //Overwrite output name
                    this.FunctionBlockDictionary[function.InputNames[0]].OutputNames = function.OutputNames.ToArray();

                    //If it is divided function update the output name of the source
                    if (SplitedFunctionDictionary.ContainsKey(function.InputNames[0]))
                    {
                        FunctionStack spliteFunction = SplitedFunctionDictionary[function.InputNames[0]];

                        for (int i = 0; i < spliteFunction.OutputNames.Length; i++)
                        {
                            if (spliteFunction.OutputNames[i] == function.InputNames[0])
                            {
                                spliteFunction.OutputNames[i] = function.OutputNames[0];

                                if (!SplitedFunctionDictionary.ContainsKey(function.OutputNames[0]))
                                {
                                    SplitedFunctionDictionary.Add(function.OutputNames[0], spliteFunction);
                                }
                            }
                        }
                    }

                    if (!(function is MultiOutputFunction) && //If the output branches, do not register and cut the link
                      !this.FunctionBlockDictionary.ContainsKey(function.OutputNames[0])) //Do not register if already registered
                    {
                        //Add link to dictionary
                        this.FunctionBlockDictionary.Add(function.OutputNames[0], this.FunctionBlockDictionary[function.InputNames[0]]);
                    }
                    else if (function is SplitFunction) //SplitFunction
                    {
                        var splitFunctions = ((SplitFunction)function).SplitedFunctions;

                        for (int i = 0; i < splitFunctions.Length; i++)
                        {
                            //Add internal FunctionStack to link dictionary
                            FunctionBlockDictionary.Add(function.OutputNames[i], splitFunctions[i]);

                            //Add to SplitFunction's list
                            SplitedFunctionDictionary.Add(function.OutputNames[i], this.FunctionBlockDictionary[function.InputNames[0]]);
                        }
                    }

                    return;
                }
            }

            //Processing for uncompressed, or MultiInput, DualInput below

            //Check if block is registered in the dictionary
            if (this.FunctionBlockDictionary.ContainsKey(function.OutputNames[0]))
            {
                //If the block has already been dictionary registered, it is concatenated to the block
                this.FunctionBlockDictionary[function.OutputNames[0]].Add(function);
            }
            else
            {
                //If it was not registered create a new block
                FunctionStack functionRecord = new FunctionStack(function, function.Name, function.InputNames, function.OutputNames);

                //Register in order of execution
                this.FunctionBlocks.Add(functionRecord);

                //Add link to dictionary
                this.FunctionBlockDictionary.Add(function.Name, functionRecord);
            }
        }

        //Forward
        public override NdArray[] Forward(params NdArray[] xs)
        {
            NdArray[] result = xs;

            //Dictionary of output data
            Dictionary<string, NdArray> outPuts = new Dictionary<string, NdArray>();

            //Register the first data in the dictionary
            for (int i = 0; i < FunctionBlocks[0].InputNames.Length; i++)
            {
                outPuts.Add(FunctionBlocks[0].InputNames[i], xs[i]);
            }

            //Run in order of registration
            for (int i = 0; i < FunctionBlocks.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[i].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                //Collect the input data
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.Add(outPuts[inputBlockNames[j]]);
                }

                //Execute function
                result = FunctionBlocks[i].Forward(inputData.ToArray());

                //Register the outputted data in the dictionary
                for (int j = 0; j < result.Length; j++)
                {
                    outPuts.Add(FunctionBlocks[i].OutputNames[j], result[j]);
                }
            }

            return result;
        }

        //Backward
        public override void Backward(params NdArray[] ys)
        {
            NdArray.Backward(ys[0]);
        }

        //Weight update process
        public override void Update()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Update();
            }
        }

        //Return specific data to the initial value after a certain process is executed
        public override void ResetState()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.ResetState();
            }
        }

        //Execute forecast
        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] result = xs;

            //Dictionary of output data
            Dictionary<string, NdArray> outPuts = new Dictionary<string, NdArray>();

            //Register the output data in the dictionary
            for (int j = 0; j < FunctionBlocks[0].InputNames.Length; j++)
            {
                outPuts.Add(FunctionBlocks[0].InputNames[j], xs[j]);
            }

            for (int i = 0; i < FunctionBlocks.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[i].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                //Collect input data
                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.Add(outPuts[inputBlockNames[j]]);
                }

                //Execute function
                result = FunctionBlocks[i].Predict(inputData.ToArray());

                //Register the output data in the dictionary
                for (int j = 0; j < result.Length; j++)
                {
                    outPuts.Add(FunctionBlocks[i].OutputNames[j], result[j]);
                }
            }

            return result;
        }

        public override void SetOptimizer(params Optimizer[] optimizers)
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.SetOptimizer(optimizers);
            }
        }
    }
}
