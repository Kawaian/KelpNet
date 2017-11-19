using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using KelpNet.Common.Functions;
using KelpNet.Functions.Mathmetrics.BasicMath;

namespace KelpNet.Common
{
    [Serializable]
    [DebuggerDisplay("{Name + ToString(\"Size\")}", Type = "{\"NdArray\" + ToString(\"Size\")}")]
    public class NdArray
    {
        public string Name = "NdArray";

        public Real[] Data;

        [NonSerialized]
        public Real[] Grad;

        //Size of each dimension of this NdArray
        public int[] Shape { private set; get; }

        //Length calculated from Shape is different from Length of Data
        public int Length { private set; get; }

        //Count the number of times used by the function to try the timing of the backward operation
        [NonSerialized]
        public int UseCount = 0;

        //If it is generated from a function, save that function here
        [NonSerialized]
        public Function ParentFunc;

        //Indicates the number of batches executed together in each function and used in the discount in the Loss function
        public int BatchCount;

        //Count the number of Backwards executed without updating and use it when executing Optimizer
        [NonSerialized]
        public int TrainCount;

        public NdArray(Array data, Function parentFunc = null)
        {
            Real[] resultData = Real.GetArray(data);

            int[] resultShape = new int[data.Rank];

            for (int i = 0; i < data.Rank; i++)
            {
                resultShape[i] = data.GetLength(i);
            }

            this.Data = resultData;
            this.Shape = resultShape;
            this.Length = Data.Length;
            this.Grad = new Real[this.Length];
            this.BatchCount = 1;
            this.TrainCount = 0;
            this.ParentFunc = parentFunc;
        }

        public NdArray(params int[] shape)
        {
            this.Data = new Real[ShapeToArrayLength(shape)];
            this.Shape = shape.ToArray();
            this.Length = Data.Length;
            this.BatchCount = 1;
            this.Grad = new Real[this.Length];
            this.TrainCount = 0;
        }

        public NdArray(Real[] data, int[] shape, int batchCount = 1, Function parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
            this.Grad = new Real[this.Length * batchCount];
            this.TrainCount = 0;
            this.ParentFunc = parentFunc;
        }

        public NdArray(int[] shape, int batchCount, Function parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new Real[this.Length * batchCount];
            this.Grad = new Real[this.Length * batchCount];
            this.TrainCount = 0;
            this.ParentFunc = parentFunc;
        }

        //Register array array as batch
        public static NdArray FromArrays(Array[] arrays, Function parentFunc = null)
        {
            int[] resultShape = new int[arrays[0].Rank];

            for (int i = 0; i < arrays[0].Rank; i++)
            {
                resultShape[i] = arrays[0].GetLength(i);
            }

            int length = arrays[0].Length;
            Real[] result = new Real[length * arrays.Length];

            for (int i = 0; i < arrays.Length; i++)
            {
                Array.Copy(Real.GetArray(arrays[i]), 0, result, length * i, length);
            }

            return new NdArray(result, resultShape, arrays.Length, parentFunc);
        }

        public static NdArray Convert(Real[] data, int[] shape, int batchCount, Function parentFunc = null)
        {
            return new NdArray(shape, batchCount, parentFunc) { Data = data };
        }

        public static NdArray ZerosLike(NdArray baseArray)
        {
            return new NdArray(baseArray.Shape.ToArray(), baseArray.BatchCount);
        }

        //Because the indexer is not so early, I do not recommend using it when accessing frequently.   Please divide it for debugging purpose.
        public Real this[int batchcount, params int[] indices]
        {
            get
            {
                return this.Data[this.GetLocalIndex(batchcount, indices)];
            }
            set
            {
                this.Data[this.GetLocalIndex(batchcount, indices)] = value;
            }
        }


        public void Reshape(params int[] shape)
        {
            int val = 0;
            int dimension = Length;

            //Calculate -1 designation
            if (shape.Contains(-1))
            {
                int minusIndex = -1;

                for (int i = 0; i < shape.Length; i++)
                {
                    if (shape[i] != -1)
                    {
                        val += Length % shape[i];

                        if (val == 0)
                        {
                            dimension /= shape[i];
                        }
                        else
                        {
                            throw new Exception("The specification of the element is wrong");
                        }
                    }
                    else
                    {
                        if (minusIndex != -1)
                        {
                            throw new Exception("Two or more -1 are specified");
                        }

                        minusIndex = i;
                    }
                }

                shape[minusIndex] = dimension;
            }
#if DEBUG
            else if (Length != ShapeToArrayLength(shape)) throw new Exception("The size of the specified Shape is not equal to the current Data.Length");
#endif

            Shape = shape.ToArray();
        }

        //Dispatch the unified array in batch and discharge
        public NdArray[] DivideArrays()
        {
            NdArray[] result = new NdArray[BatchCount];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = GetSingleArray(i);
            }

            return result;
        }

        //Eject the array corresponding to the batch number
        public NdArray GetSingleArray(int i)
        {
            Real[] data = new Real[this.Length];
            Array.Copy(this.Data, i * this.Length, data, 0, this.Length);

            return new NdArray(data, this.Shape);
        }

        static int ShapeToArrayLength(params int[] shapes)
        {
            int result = 1;

            foreach (int shape in shapes)
            {
                result *= shape;
            }

            return result;
        }

        public void Backward()
        {
            if (ParentFunc != null)
            {
                for (int i = 0; i < Grad.Length; i++)
                {
                    Grad[i] = 1;
                }

                NdArray.Backward(this);
            }
        }

        public static void Backward(NdArray y)
        {
            if (y.ParentFunc != null)
            {
                List<NdArray[]> prevInputs = y.ParentFunc.PrevInputs;
                NdArray[] xs = prevInputs[prevInputs.Count - 1];

                y.ParentFunc.Backward(y);

                for (int i = 0; i < xs.Length; i++)
                {
                    if (xs[i].UseCount == 0)
                    {
                        NdArray.Backward(xs[i]);
                    }
                }
            }
        }

        public void CountUp()
        {
            TrainCount++;
        }

        //Correction of slope
        public bool Reduce()
        {
            if (this.TrainCount > 0)
            {
                for (int i = 0; i < this.Grad.Length; i++)
                {
                    this.Grad[i] /= this.TrainCount;
                }

                return true;
            }

            return false;
        }

        //Initialization of slope
        public void ClearGrad()
        {
            this.Grad = new Real[this.Data.Length];

            //Reset counter
            this.TrainCount = 0;
        }

        public override string ToString()
        {
            return ToString(this.Data);
        }

        public string ToString(string format)
        {
            switch (format)
            {
                case "Data":
                    return ToString(this.Data);

                case "Grad":
                    return ToString(this.Grad);

                case "Shape":
                    return "[" + string.Join(",", Shape) + "]";

                case "Size":
                    return "[" + string.Join(",", Shape) + "]" +
                           (BatchCount > 1 ? "x" + BatchCount + "batch" : string.Empty);

                case "Name":
                    return Name;

                default:
                    return Name;
            }
        }

        public string ToString(Real[] datas)
        {
            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0; //Maximum value of integer part
            int realMaxLength = 0; //Maximum value after the decimal point
            bool isExponential = false; //Will it be exponential representation?

            foreach (Real data in datas)
            {
                string[] divStr = ((double)data).ToString().Split('.');
                intMaxLength = Math.Max(intMaxLength, divStr[0].Length);

                if (divStr.Length > 1)
                {
                    isExponential |= divStr[1].Contains("E");
                }

                if (realMaxLength != 8 && divStr.Length == 2)
                {
                    realMaxLength = Math.Max(realMaxLength, divStr[1].Length);

                    if (realMaxLength > 8)
                    {
                        realMaxLength = 8;
                    }
                }
            }

            //Get divisor of array
            int[] commonDivisorList = new int[this.Shape.Length];

            //First manual acquisition
            commonDivisorList[0] = this.Shape[this.Shape.Length - 1];

            for (int i = 1; i < this.Shape.Length; i++)
            {
                commonDivisorList[i] = commonDivisorList[i - 1] * this.Shape[this.Shape.Length - i - 1];
            }

            if (this.BatchCount > 1)
            {
                sb.Append("{");
            }

            for (int batch = 0; batch < this.BatchCount; batch++)
            {
                int indexOffset = batch * Length;
                //Leading parenthesis
                for (int i = 0; i < this.Shape.Length; i++)
                {
                    sb.Append("[");
                }

                int closer = 0;
                for (int i = 0; i < Length; i++)
                {
                    string[] divStr;
                    if (isExponential)
                    {
                        divStr = string.Format("{0:0.00000000e+00}", (Real)datas[indexOffset + i]).Split('.');
                    }
                    else
                    {
                        divStr = ((Real)datas[indexOffset + i]).ToString().Split('.');
                    }

                    //Align indentation with maximum number of characters
                    for (int j = 0; j < intMaxLength - divStr[0].Length; j++)
                    {
                        sb.Append(" ");
                    }
                    sb.Append(divStr[0]);
                    if (realMaxLength != 0)
                    {
                        sb.Append(".");
                    }
                    if (divStr.Length == 2)
                    {
                        sb.Append(divStr[1].Length > 8 && !isExponential ? divStr[1].Substring(0, 8) : divStr[1]);
                        for (int j = 0; j < realMaxLength - divStr[1].Length; j++)
                        {
                            sb.Append(" ");
                        }
                    }
                    else
                    {
                        for (int j = 0; j < realMaxLength; j++)
                        {
                            sb.Append(" ");
                        }
                    }

                    //If it is perfect after investigating divisors, it outputs parentheses
                    if (i != Length - 1)
                    {
                        foreach (int commonDivisor in commonDivisorList)
                        {
                            if ((i + 1) % commonDivisor == 0)
                            {
                                sb.Append("]");
                                closer++;
                            }
                        }

                        sb.Append(" ");

                        if ((i + 1) % commonDivisorList[0] == 0)
                        {
                            //Add line feed by closing parenthesis
                            for (int j = 0; j < closer; j++)
                            {
                                sb.Append("\n");
                            }
                            closer = 0;

                            if (BatchCount > 1) sb.Append(" ");

                            //Indentation before bracket
                            foreach (int commonDivisor in commonDivisorList)
                            {
                                if ((i + 1) % commonDivisor != 0)
                                {
                                    sb.Append(" ");
                                }
                            }
                        }

                        foreach (int commonDivisor in commonDivisorList)
                        {
                            if ((i + 1) % commonDivisor == 0)
                            {
                                sb.Append("[");
                            }
                        }
                    }
                }

                //Parenthesis at the back end
                for (int i = 0; i < this.Shape.Length; i++)
                {
                    sb.Append("]");
                }

                if (batch < this.BatchCount - 1)
                {
                    sb.Append("},\n{");
                }
            }

            if (this.BatchCount > 1)
            {
                sb.Append("}");
            }

            return sb.ToString();
        }


        public static NdArray operator +(NdArray a, NdArray b)
        {
            return new Add().Forward(a, b)[0];
        }

        public static NdArray operator +(NdArray a, Real b)
        {
            return new AddConst().Forward(a, b)[0];
        }

        public static NdArray operator +(Real a, NdArray b)
        {
            return new AddConst().Forward(b, a)[0];
        }


        public static NdArray operator *(NdArray a, NdArray b)
        {
            return new Mul().Forward(a, b)[0];
        }

        public static NdArray operator *(NdArray a, Real b)
        {
            return new MulConst().Forward(a, b)[0];
        }

        public static NdArray operator *(Real a, NdArray b)
        {
            return new MulConst().Forward(b, a)[0];
        }


        public static NdArray operator -(NdArray a, NdArray b)
        {
            return new Sub().Forward(a, b)[0];
        }

        public static NdArray operator -(NdArray a, Real b)
        {
            return new SubConst().Forward(a, b)[0];
        }

        public static NdArray operator -(Real a, NdArray b)
        {
            return new ConstSub().Forward(a, b)[0];
        }


        public static NdArray operator /(NdArray a, NdArray b)
        {
            return new Div().Forward(a, b)[0];
        }

        public static NdArray operator /(NdArray a, Real b)
        {
            return new DivConst().Forward(a, b)[0];
        }

        public static NdArray operator /(Real a, NdArray b)
        {
            return new ConstDiv().Forward(a, b)[0];
        }

        public static implicit operator NdArray(Real[] a)
        {
            return new NdArray(a);
        }

        public static implicit operator NdArray(Real a)
        {
            return new NdArray(new[] { a });
        }

        public static implicit operator NdArray(long a)
        {
            return new NdArray(new[] { (Real)a });
        }

        //Method to create copy
        public NdArray Clone()
        {
            return new NdArray
            {
                ParentFunc = ParentFunc,
                Data = Data.ToArray(),
                Grad = Grad.ToArray(),
                Shape = Shape.ToArray(),
                Name = Name,
                Length = Length,
                BatchCount = BatchCount,
                UseCount = UseCount,
                TrainCount = TrainCount
            };
        }

        public static NdArray Sum(NdArray a, bool keepDims = false, params int[] axis)
        {
#if DEBUG
            if (axis.Length != axis.Distinct().ToArray().Length)
            {
                throw new Exception("The specified element is duplicated");
            }

            if (axis.Length != 0 && a.Shape.Length < axis.Max())
            {
                throw new Exception("The specified element is out of range");
            }
#endif
            if (axis.Length == 0)
            {
                axis = Enumerable.Range(0, a.Shape.Length).ToArray();
            }

            Array.Sort(axis);

            NdArray result = Sum(a, axis[0]);

            for (int i = 1; i < axis.Length; i++)
            {
                result = Sum(result, axis[i] - i);
            }

            if (keepDims)
            {
                List<int> resultKeepDimShape = new List<int>();
                int count = a.Shape.Length - result.Shape.Length;

                for (int i = 0; i < count; i++)
                {
                    resultKeepDimShape.Add(1);
                }

                resultKeepDimShape.AddRange(result.Shape);
                result.Shape = resultKeepDimShape.ToArray();
            }

            return result;
        }

        private static NdArray Sum(NdArray a, int axis)
        {
            int[] resultShape = new int[a.Shape.Length - 1];

            for (int i = 0, j = 0; i < a.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = a.Shape[i];
                }
            }

            NdArray result = new NdArray(resultShape, a.BatchCount);

            for (int i = 0; i < a.Length; i++)
            {
                List<int> index = new List<int>(a.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < a.BatchCount; batchCount++)
                {
                    result.Data[batchCount * result.Length + localIndex] += a.Data[batchCount * a.Length + i];
                    result.Grad[batchCount * result.Length + localIndex] += a.Grad[batchCount * a.Length + i];
                }
            }

            return result;
        }

        public static NdArray[] Split(NdArray array, int indices, int axis = 1)
        {
            return Split(array, new[] { indices }, axis);
        }

        public static NdArray[] Split(NdArray array, int[] indices, int axis = 1)
        {
            int[] shapeOffets = new int[indices.Length + 1];        //An array in which the leading 0 of the input indices is added
            int[] resultAxisShapes = new int[indices.Length + 1];   //Shape of specified axis after division

            for (int i = 0; i < indices.Length; i++)
            {
                shapeOffets[i + 1] = indices[i];
                resultAxisShapes[i] = indices[i] - shapeOffets[i];
            }
            resultAxisShapes[indices.Length] = array.Shape[axis] - indices[indices.Length - 1];

            NdArray[] resultArrays = new NdArray[indices.Length + 1];

            for (int i = 0; i < resultArrays.Length; i++)
            {
                int[] resultShape = array.Shape.ToArray();
                resultShape[axis] = resultAxisShapes[i];
                resultArrays[i] = new NdArray(resultShape, array.BatchCount);
            }

            for (int batchCount = 0; batchCount < array.BatchCount; batchCount++)
            {
                for (int i = 0; i < resultArrays.Length; i++)
                {
                    for (int j = 0; j < resultArrays[i].Length; j++)
                    {
                        int[] resultIndex = resultArrays[i].GetDimensionsIndex(j);
                        resultIndex[axis] += shapeOffets[i];
                        int localIndex = array.GetLocalIndex(batchCount, resultIndex);

                        resultArrays[i].Data[batchCount * resultArrays[i].Length + j] = array.Data[localIndex];
                        resultArrays[i].Grad[batchCount * resultArrays[i].Length + j] = array.Grad[localIndex];
                    }
                }
            }

            return resultArrays;
        }

        public static NdArray Concatenate(NdArray a, NdArray b, int axis)
        {
            int[] shapeList = a.Shape.ToArray();
            shapeList[axis] += b.Shape[axis];

#if DEBUG
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (i != axis && a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("The size of the array is not matched");
                }
            }

            if (a.BatchCount != b.BatchCount)
            {
                throw new Exception("Batch size is not matched");
            }
#endif

            NdArray result = new NdArray(shapeList.ToArray(), a.BatchCount);

            for (int batchCount = 0; batchCount < a.BatchCount; batchCount++)
            {
                int aInputBatchoffset = batchCount * a.Length;
                int bInputBatchoffset = batchCount * b.Length;

                for (int i = 0; i < a.Length; i++)
                {
                    int resultindex = result.GetLocalIndex(batchCount, a.GetDimensionsIndex(i));

                    result.Data[resultindex] = a.Data[i + aInputBatchoffset];
                    result.Grad[resultindex] = a.Grad[i + aInputBatchoffset];
                }

                for (int i = 0; i < b.Length; i++)
                {
                    int[] tmpIndex = b.GetDimensionsIndex(i);
                    tmpIndex[axis] += a.Shape[axis];

                    int resultIndex = result.GetLocalIndex(batchCount, tmpIndex);

                    result.Data[resultIndex] = b.Data[i + bInputBatchoffset];
                    result.Grad[resultIndex] = b.Grad[i + bInputBatchoffset];
                }
            }

            return result;
        }

        internal int[] GetDimensionsIndex(int index)
        {
            //Correct batch
            int batchCount = index / this.Length;
            index -= this.Length * batchCount;

            int[] dimensionsIndex = new int[this.Shape.Length];

            for (int i = this.Shape.Length - 1; i >= 0; i--)
            {
                dimensionsIndex[i] = index % this.Shape[i];
                index /= this.Shape[i];
            }

            return dimensionsIndex;
        }

        internal int GetLocalIndex(int batchIndex, params int[] indices)
        {
            int indicesLastIndex = indices.Length - 1;
            int index = batchIndex * this.Length + indices[indicesLastIndex];
            int rankoffset = 1;

            for (int i = 1; i < indices.Length; i++)
            {
                rankoffset *= this.Shape[indicesLastIndex--];
                index += indices[indicesLastIndex] * rankoffset;
            }

            return index;
        }
    }
}