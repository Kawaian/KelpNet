﻿using Cloo;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KelpNet.Common
{
    public interface IDeviceReal : IDisposable
    {
        Real this[int index] { get; set; }
        bool IsGpu { get; }
        Real[] ToArray();
        void CopyTo(int sourceOffset, Real[] target, int destOffset, int length);
        void CopyTo(int sourceOffset, IDeviceReal target, int destOffset, int length);
        IDeviceReal Clone();
    }

    public class CpuRealArray : IDeviceReal
    {
        public Real this[int index] { get => buffer[index]; set => buffer[index]=value; }

        public bool IsGpu => false;

        public Real[] buffer;

        public CpuRealArray(Real[] data)
        {
            buffer = data;
        }

        public void Dispose()
        {
            buffer = null;
        }

        public Real[] ToArray()
        {
            return buffer.ToArray();
        }

        public void CopyTo(int sourceOffset, Real[] target, int destOffset, int length)
        {
            Array.Copy(buffer, sourceOffset, target, destOffset, length);
        }

        public void CopyTo(int sourceOffset, IDeviceReal target, int destOffset, int length)
        {
            if (target.IsGpu)
            {
                var t = (GpuRealArray)target;
                Weaver.CommandQueue.WriteToBuffer(buffer, t.buffer, true, sourceOffset, destOffset, length, null);
            }
            else
            {
                var t = (CpuRealArray)target;
                CopyTo(sourceOffset, t.buffer, destOffset, length);
            }
        }

        public IDeviceReal Clone()
        {
            return new CpuRealArray((Real[])buffer.Clone());
        }
    }

    public class GpuRealArray : IDeviceReal
    {
        public Real this[int index]
        {
            get
            {
                Real[] real = new Real[1];
                Weaver.CommandQueue.ReadFromBuffer(buffer, ref real, true, index, 0, 1, null);
                return real[0];
            }
            set
            {
                Weaver.CommandQueue.WriteToBuffer(new[] { value }, buffer, true, 0, index, 1, null);
            }
        }

        public bool IsGpu => true;
        public int Length { private set; get; }

        public ComputeBuffer<Real> buffer;

        public GpuRealArray(Real[] data, ComputeMemoryFlags flag)
        {
            Length = data.Length;
            buffer = new ComputeBuffer<Real>(Weaver.Context, flag, data);
        }

        public GpuRealArray(ComputeBuffer<Real> data)
        {
            Length = (int)data.Count;
            buffer = data;
        }

        public void Dispose()
        {
            if (buffer != null)
            {
                buffer.Dispose();
                buffer = null;
            }
        }

        public Real[] ToArray()
        {
            var fetch = new Real[Length];
            Weaver.CommandQueue.ReadFromBuffer(buffer, ref fetch, true, null);
            return fetch;
        }

        public void CopyTo(int sourceOffset, Real[] target, int destOffset, int length)
        {
            Weaver.CommandQueue.ReadFromBuffer(buffer, ref target, true, sourceOffset, destOffset, length, null);
        }

        public void CopyTo(int sourceOffset, IDeviceReal target, int destOffset, int length)
        {
            if (target.IsGpu)
            {
                var t = (GpuRealArray)target;
                Weaver.CommandQueue.CopyBuffer(buffer, t.buffer, sourceOffset, destOffset, length, null);
            }
            else
            {
                var t = (CpuRealArray)target;
                Weaver.CommandQueue.WriteToBuffer(t.buffer, buffer, true, sourceOffset, destOffset, length, null);
            }
        }

        public IDeviceReal Clone()
        {
            var newBuf = new ComputeBuffer<Real>(Weaver.Context, buffer.Flags, buffer.Count);
            Weaver.CommandQueue.CopyBuffer(buffer, newBuf, null);
            return new GpuRealArray(newBuf);
        }
    }

    public class RealArray : IDisposable
    {
        public static explicit operator RealArray(Real[] data)
        {
            var temp = new RealArray(data, isGpu : false);

            return temp;
        }

        public static explicit operator ComputeBuffer<Real>(RealArray data)
        {
            return data.AsBuffer();
        }

        public static explicit operator Real[](RealArray data)
        {
            return data.AsArray();
        }

        public static void Copy(Real[] source, RealArray target)
        {
            Copy(source, target, target.Length);
        }

        public static void Copy(Real[] source, RealArray target, int length)
        {
            Copy(source, 0, target, 0, length);
        }

        public static void Copy(Real[] source, int sourceOffset, RealArray target, int targetOffset, int length)
        {
            ((RealArray)source).CopyTo(sourceOffset, target, targetOffset, length);
        }

        public static void Copy(RealArray source, int sourceOffset, RealArray target, int targetOffset, int length)
        {
            source.CopyTo(sourceOffset, target, targetOffset, length);
        }

        public static void Copy(RealArray source, RealArray target, int length)
        {
            source.CopyTo(target, length);
        }

        public static void Copy(RealArray source, RealArray target)
        {
            source.CopyTo(target, target.Length);
        }

        public static void Copy(RealArray source, int sourceOffset, Real[] target, int targetOffset, int length)
        {
            source.CopyTo(sourceOffset, target, targetOffset, length);
        }

        public static void Copy(RealArray source, Real[] target, int length)
        {
            source.CopyTo(target, length);
        }

        public static void Copy(RealArray source, Real[] target)
        {
            source.CopyTo(target, target.Length);
        }

        public const ComputeMemoryFlags DefaultFlag = ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer;

        public Real this[int index]
        {
            get => bank[index];
            set => bank[index] = value;
        }

        public int Length => Count;
        public int Count { private set; get; }

        bool isDisposed = false;
        public bool IsDisposed => isDisposed;
        public bool IsGpu => bank == null ? false : bank.IsGpu;

        public ComputeMemoryFlags Flag { get; set; }

        IDeviceReal bank;
        StackTrace creationStack;
        StackTrace toGpuStack;
        StackTrace toCpuStack;

        public RealArray(int length, bool isGpu = false, ComputeMemoryFlags flag = DefaultFlag) : this(new Real[length], isGpu, flag)
        {

        }

        public RealArray(Real[] data, bool isGpu = false, ComputeMemoryFlags flag = DefaultFlag)
        {
            Flag = flag;
            Count = data.Length;
            if (isGpu)
            {
                bank = new GpuRealArray(data, flag);
            }
            else
            {
                bank = new CpuRealArray(data);
            }
            
#if DEBUG
            if(isGpu)
                Console.WriteLine("Create GPU Buffer");
            creationStack = new StackTrace(true);
#endif
        }

        public RealArray(IDeviceReal bank, int length, ComputeMemoryFlags flags = DefaultFlag)
        {
            this.bank = bank;
            Count = length;
            Flag = flags;
            if (IsGpu)
            {
                Flag = ((GpuRealArray)bank).buffer.Flags;
            }
#if DEBUG
            if (IsGpu)
                Console.WriteLine("Create from GPU Buffer");
            creationStack = new StackTrace(true);
#endif
        }

        public void CopyTo(RealArray traget)
        {
            CopyTo(traget, Length);
        }

        public void CopyTo(RealArray target, int length)
        {
            CopyTo(0, target, 0, length);
        }

        public void CopyTo(int sourceOffset, RealArray target, int destOffset, int length)
        {
            bank.CopyTo(sourceOffset, target.bank, destOffset, length);
        }

        public void CopyTo(Real[] target)
        {
            CopyTo(target, target.Length);
        }

        public void CopyTo(Real[] target, int length)
        {
            CopyTo(0, target, 0, length);
        }

        public void CopyTo(int sourceOffset, Real[] target, int targetOffset, int length)
        {
            bank.CopyTo(sourceOffset, target, targetOffset, length);
        }

        /// <summary>
        /// Cast RealArray to Real[]. It try to cast if possible (in cpu)
        /// </summary>
        /// <returns></returns>
        public Real[] AsArray()
        {
            return !bank.IsGpu ? ((CpuRealArray)bank).buffer : throw new InvalidCastException("Cant cast GPU buffer to CPU array");
        }

        public ComputeBuffer<Real> AsBuffer()
        {
            return bank.IsGpu ? ((GpuRealArray)bank).buffer : throw new InvalidCastException("Cant cast CPU array to GPU buffer");
        }

        /// <summary>
        /// Convert RealArray to Real[]. It allocate new memory.
        /// </summary>
        /// <returns></returns>
        public Real[] ToArray()
        {
            return bank.ToArray();
        }

        IDeviceReal preBank;
        public static bool KeepAllocation = true;

        public void ToGpu()
        {
            ToGpu(!KeepAllocation);
        }

        public void ToGpu(bool dispose)
        {
            ToGpu(Flag, dispose);
        }

        public void ToGpu(ComputeMemoryFlags flag, bool dispose)
        {
            if (!IsGpu)
            {
#if DEBUG
                Console.WriteLine("Convert To GPU");
                toGpuStack = new StackTrace(true);
#endif
                Flag = flag;
                Real[] buffer = ((CpuRealArray)bank).buffer;
                if (dispose)
                {
                    if (preBank != null)
                    {
                        preBank.Dispose();
                        preBank = null;
                    }

                    bank.Dispose();
                    bank = new GpuRealArray(buffer, flag);
                }
                else
                {
                    if (preBank != null)
                    {
                        bank.CopyTo(0, preBank, 0, Length);
                    }
                    else
                    {
                        preBank = new GpuRealArray(buffer, flag);
                    }

                    var temp = preBank;
                    preBank = bank;
                    bank = temp;
                }
#if DEBUG
                if (!(bank is GpuRealArray))
                {
                    Console.WriteLine("Wrong preBank device");
                    ShowTrace();
                }
#endif
            }
        }

        public void ToCpu()
        {
            ToCpu(!KeepAllocation);
        }

        public void ToCpu(bool dispose)
        {
            if (IsGpu)
            {
#if DEBUG
                Console.WriteLine("Convert To CPU");
                toCpuStack = new StackTrace(true);
#endif
                if (dispose)
                {
                    if(preBank != null)
                    {
                        preBank.Dispose();
                        preBank = null;
                    }

                    Real[] buffer = bank.ToArray();
                    bank.Dispose();
                    bank = new CpuRealArray(buffer);
                }
                else
                {
                    if (preBank != null)
                    {
                        bank.CopyTo(0, preBank, 0, Length);
                    }
                    else
                    {
                        preBank = new CpuRealArray(bank.ToArray());
                    }

                    var temp = preBank;
                    preBank = bank;
                    bank = temp;
                }
#if DEBUG
                if (!(bank is CpuRealArray))
                {
                    Console.WriteLine("Wrong preBank device");
                    ShowTrace();
                }
#endif
            }
        }

        public RealArray Clone()
        {
            return new RealArray(bank.Clone(), Length, Flag);
        }

        ~RealArray()
        {
            OnDispose();
        }

        private void OnDispose(bool onUserDispose = false)
        {
            if (!isDisposed)
            {
#if DEBUG
                if (IsGpu)
                { 
                    Console.WriteLine("Dispose Gpu Buffer");

                    if (!onUserDispose)
                    {
                        Console.WriteLine($"Information: Gpu array is not disposed in code properly. RealArray[{Length}]");
                        ShowTrace();
                    }
                }
#endif

                isDisposed = true;

                try
                {
                    if (bank != null)
                    {
                        bank.Dispose();
                        bank = null;
                    }

                    if(preBank != null)
                    {
                        preBank.Dispose();
                        preBank = null;
                    }
                }
                catch (Exception ex)
                {
                    string detail;
                    try
                    {
                        Show("Buffer Detail", $"Bank:{bank}, Length:{Length}");
                    }
                    catch (Exception dex)
                    {
                        detail = "Error while get detail message\n" + dex.ToString();
                    }
                    Show("Exception", ex);
                    ShowTrace();
                    throw new Exception("Dispose Error", ex);
                }
            }
        }

        public void ShowTrace()
        {
            Show("Create Stack", creationStack);
            Show("ToGpuStack", toGpuStack);
            Show("ToCpuStack", toCpuStack);
        }

        private void Show(string title, object content)
        {
            Console.WriteLine($"========== {title} ===========");
            Console.WriteLine(content);
        }

        public void Dispose()
        {
            OnDispose(true);
        }
    }
}