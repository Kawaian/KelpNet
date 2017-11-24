using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class StochasticDepth : Function //Use it to replace SplitFunction
    {
        const string FUNCTION_NAME = "StochasticDepth";

        private readonly Real _pl;

        private readonly List<bool> _skipList = new List<bool>();

        private readonly Function _function; //Skipped by probability
        private readonly Function _resBlock; //Always executed

        private bool IsSkip()
        {
            bool result = Mother.Dice.NextDouble() >= this._pl;

            this._skipList.Add(result);

            return result;
        }

        public StochasticDepth(Function function, Function resBlock = null, double pl = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._function = function;
            this._resBlock = resBlock;

            this._pl = pl;
        }

        protected override NdArray[] OnForward(params NdArray[] xs)
        {
            List<NdArray> resultArray = new List<NdArray>();
            NdArray[] resResult = xs;

            if (_resBlock != null)
            {
                resResult = _resBlock.Forward(xs);
            }

            resultArray.AddRange(resResult);

            if (!IsSkip())
            {
                Real scale = 1 / (1 - this._pl);
                NdArray[] result = _function.Forward(xs);

                for (int i = 0; i < result.Length; i++)
                {
                    for (int j = 0; j < result[i].Data.Length; j++)
                    {
                        result[i].Data[j] *= scale;
                    }
                }

                resultArray.AddRange(result);
            }
            else
            {
                NdArray[] result = new NdArray[resResult.Length];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray(resResult[i].Shape, resResult[i].BatchCount, resResult[i].ParentFunc);
                }

                resultArray.AddRange(result);
            }

            return resultArray.ToArray();
        }

        protected override void OnBackward(params NdArray[] ys)
        {
            if (_resBlock != null)
            {
                _resBlock.Backward(ys);
            }

            bool isSkip = this._skipList[this._skipList.Count - 1];
            this._skipList.RemoveAt(this._skipList.Count - 1);

            if (!isSkip)
            {
                NdArray[] copyys = new NdArray[ys.Length];

                Real scale = 1 / (1 - this._pl);

                for (int i = 0; i < ys.Length; i++)
                {
                    copyys[i] = ys[i].Clone();

                    for (int j = 0; j < ys[i].Data.Length; j++)
                    {
                        copyys[i].Data[j] *= scale;
                    }
                }

                _function.Backward(copyys);
            }
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return _function.Predict(xs);
        }
    }
}
