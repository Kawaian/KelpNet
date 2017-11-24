using System;
using System.Runtime.InteropServices;
//using RealType = System.Double;
using RealType = System.Single;

namespace KelpNet.Common
{
    class RealTool
    {
        [DllImport("kernel32.dll")]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);
    }

    /// <summary>
    /// Default calculation unit. Single or Double.
    /// </summary>
    [Serializable]
    public struct Real : IComparable<Real>
    {
        public readonly RealType Value;

        private Real(double value)
        {
            this.Value = (RealType)value;
        }

        public static implicit operator Real(double value)
        {
            return new Real(value);
        }

        public static implicit operator RealType(Real real)
        {
            return real.Value;
        }

        public int CompareTo(Real other)
        {
            return this.Value.CompareTo(other.Value);
        }

        public override string ToString()
        {
            return this.Value.ToString();
        }

        public static Real[] GetArray(Array data)
        {
            Type arrayType = data.GetType().GetElementType();
            Real[] resultData = new Real[data.Length];

            //Absorption of type mismatch here
            if (arrayType != typeof(RealType) && arrayType != typeof(Real))
            {
                //Prepare one-dimensional length array
                Array array = Array.CreateInstance(arrayType, data.Length);
                //Make it one-dimensional
                Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * resultData.Length);

                data = new RealType[array.Length];

                //Copy while converting type
                Array.Copy(array, data, array.Length);
            }

            //Strike data
            int size = Marshal.SizeOf(typeof(RealType)) * data.Length;
            GCHandle gchObj = GCHandle.Alloc(data, GCHandleType.Pinned);
            GCHandle gchBytes = GCHandle.Alloc(resultData, GCHandleType.Pinned);
            RealTool.CopyMemory(gchBytes.AddrOfPinnedObject(), gchObj.AddrOfPinnedObject(), size);
            gchObj.Free();
            gchBytes.Free();

            return resultData;
        }
    }
}
