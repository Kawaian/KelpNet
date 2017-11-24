using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Mathmetrics;
using KelpNet.Functions.Normalization;

namespace ChainerModelLoader
{
    public class ChainerModelDataLoader
    {
        public static void ModelLoad(string path, FunctionStack model)
        {
            var modelData = new NpzDictionary(path);

            foreach (var function in model.Functions)
            {
                SetParams(function, modelData);
            }
        }

        static void SetParams(Function func, NpzDictionary modelData)
        {
            if (func is Linear)
            {
                Linear linear = (Linear)func;

                RealArray.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), linear.Weight.Data, linear.Weight.Data.Length);

                if (!linear.NoBias)
                {
                    RealArray.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), linear.Bias.Data, linear.Bias.Data.Length);
                }
            }
            else if (func is Convolution2D)
            {
                Convolution2D conv2D = (Convolution2D)func;

                RealArray.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), conv2D.Weight.Data, conv2D.Weight.Data.Length);

                if (!conv2D.NoBias)
                {
                    RealArray.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), conv2D.Bias.Data, conv2D.Bias.Data.Length);
                }
            }
            else if (func is Deconvolution2D)
            {
                Deconvolution2D deconv2D = (Deconvolution2D)func;

                RealArray.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), deconv2D.Weight.Data, deconv2D.Weight.Data.Length);

                if (!deconv2D.NoBias)
                {
                    RealArray.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), deconv2D.Bias.Data, deconv2D.Bias.Data.Length);
                }
            }
            else if (func is EmbedID)
            {
                EmbedID embed = (EmbedID)func;

                RealArray.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), embed.Weight.Data, embed.Weight.Data.Length);
            }
            else if (func is BatchNormalization)
            {
                BatchNormalization bn = (BatchNormalization)func;

                RealArray.Copy(Real.GetArray(modelData[func.Name + "/beta.npy"]), bn.Beta.Data, bn.Beta.Data.Length);
                RealArray.Copy(Real.GetArray(modelData[func.Name + "/gamma.npy"]), bn.Gamma.Data, bn.Gamma.Data.Length);

                if (bn.IsTrain)
                {
                    if (modelData.ContainsKey(func.Name + "/avg_mean.npy")) RealArray.Copy(Real.GetArray(modelData[func.Name + "/avg_mean.npy"]), bn.AvgMean.Data, bn.AvgMean.Data.Length);
                    if (modelData.ContainsKey(func.Name + "/avg_var.npy")) RealArray.Copy(Real.GetArray(modelData[func.Name + "/avg_var.npy"]), bn.AvgVar.Data, bn.AvgVar.Data.Length);
                }
            }
            else if (func is MultiplyScale)
            {
                MultiplyScale scale = (MultiplyScale)func;

                RealArray.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), scale.Weight.Data, scale.Weight.Data.Length);

                if (scale.BiasTerm)
                {
                    RealArray.Copy(Real.GetArray(modelData[func.Name + "/bias/b.npy"]), scale.Bias.Data, scale.Bias.Data.Length);
                }

            }
        }
    }
}
