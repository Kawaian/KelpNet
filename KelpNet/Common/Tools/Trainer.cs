using System;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Loss;

namespace KelpNet.Common.Tools
{
    //Class to perform network training
    //Mainly responsible for type conversion of Array-> NdArray
    public class Trainer
    {
        //Perform learning process in batch
        public static Real Train(FunctionStack functionStack, Array[] input, Array[] teach, LossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, NdArray.FromArrays(input), NdArray.FromArrays(teach), lossFunction, isUpdate);
        }

        //Perform learning process in batch
        public static Real Train(FunctionStack functionStack, NdArray input, NdArray teach, LossFunction lossFunction, bool isUpdate = true)
        {
            //For preserving error of result
            NdArray[] result = functionStack.Forward(input);
            Real sumLoss = lossFunction.Evaluate(result, teach);

            //Run Backward's batch
            functionStack.Backward(result);

            //update
            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        //Accuracy measurement
        public static double Accuracy(FunctionStack functionStack, Array[] x, Array[] y)
        {
            return Accuracy(functionStack, NdArray.FromArrays(x), NdArray.FromArrays(y));
        }

        public static double Accuracy(FunctionStack functionStack, NdArray x, NdArray y)
        {
            double matchCount = 0;

            NdArray forwardResult = functionStack.Predict(x)[0];

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 0; i < forwardResult.Length; i++)
                {
                    if (maxval < forwardResult.Data[i + b * forwardResult.Length])
                    {
                        maxval = forwardResult.Data[i + b * forwardResult.Length];
                        maxindex = i;
                    }
                }

                if (maxindex == (int)y.Data[b * y.Length])
                {
                    matchCount++;
                }
            }

            return matchCount / x.BatchCount;
        }
    }
}
