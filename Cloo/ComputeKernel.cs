#region License

/*

  Copyright (c) 2009 - 2013 Fatjon Sakiqi

  Permission is hereby granted, free of charge, to any person
  obtaining a copy of this software and associated documentation
  files (the "Software"), to deal in the Software without
  restriction, including without limitation the rights to use,
  copy, modify, merge, publish, distribute, sublicense, and / or sell
  copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following
  conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.

  */

#endregion

namespace Cloo
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Runtime.InteropServices;
    using System.Threading;
    using Cloo.Bindings;

    ///<summary>
    ///Represents an OpenCL kernel.
    ///</summary>
    ///<remarks> A kernel object encapsulates a specific kernel function declared in a program and the argument values ​​to be used when executing this kernel function. </remarks>
    ///<seealso cref = "ComputeCommandQueue"/>
    ///<seealso cref = "ComputeProgram"/>
    public class ComputeKernel : ComputeResource
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContext context;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string functionName;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeProgram program;

        #endregion

        #region Properties

        ///<summary>
        ///The handle of the <see cref = "ComputeKernel"/>.
        ///</summary>
        public CLKernelHandle Handle
        {
            get;
            protected set;
        }

        ///<summary>
        ///Gets the <see cref = "ComputeContext"/> associated with the <see cref = "ComputeKernel"/>.
        ///</summary>
        ///<value> The <see cref = "ComputeContext"/> associated with the <see cref = "ComputeKernel"/>. </value>
        public ComputeContext Context { get { return context; } }

        ///<summary>
        ///Gets the function name of the <see cref = "ComputeKernel"/>.
        ///</summary>
        ///<value> The function name of the <see cref = "ComputeKernel"/>. </value>
        public string FunctionName { get { return functionName; } }

        ///<summary>
        ///Gets the <see cref = "ComputeProgram"/> that the <see cref = "ComputeKernel"/> belongs to.
        ///</summary>
        ///<value> The <see cref = "ComputeProgram"/> that the <see cref = "ComputeKernel"/> belongs to. </value>
        public ComputeProgram Program { get { return program; } }

        #endregion

        #region Constructors

        internal ComputeKernel(CLKernelHandle handle, ComputeProgram program)
        {
            Handle = handle;
            SetID(Handle.Value);

            context = program.Context;
            functionName = GetStringInfo<CLKernelHandle, ComputeKernelInfo>(Handle, ComputeKernelInfo.FunctionName, CL12.GetKernelInfo);
            this.program = program;

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        internal ComputeKernel(string functionName, ComputeProgram program)
        {
            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL12.CreateKernel(program.Handle, functionName, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

            context = program.Context;
            this.functionName = functionName;
            this.program = program;

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        #endregion

        #region Public methods

        ///<summary>
        ///Gets the amount of local memory in bytes used by the <see cref = "ComputeKernel"/>.
        ///</summary>
        ///<param name = "device"> One of the <see cref = "ComputeKernel.Program.Device"/> s. </param>
        ///<returns> The amount of local memory in bytes used by the <see cref = "ComputeKernel"/>. </returns>
        public long GetLocalMemorySize(ComputeDevice device)
        {
            return GetInfo<CLKernelHandle, CLDeviceHandle, ComputeKernelWorkGroupInfo, long>(
                Handle, device.Handle, ComputeKernelWorkGroupInfo.LocalMemorySize, CL12.GetKernelWorkGroupInfo);
        }

        ///<summary>
        ///Gets the compile work-group size specified by the <c> __ attribute __ ((reqd_work_group_size (X, Y, Z))) </c> qualifier.
        ///</summary>
        ///<param name = "device"> One of the <see cref = "ComputeKernel.Program.Device"/> s. </param>
        ///(0, 0, 0) is returned (0, 0, 0) is returned  . </returns>
        public long[] GetCompileWorkGroupSize(ComputeDevice device)
        {
            return ComputeTools.ConvertArray(
                GetArrayInfo<CLKernelHandle, CLDeviceHandle, ComputeKernelWorkGroupInfo, IntPtr>(
                    Handle, device.Handle, ComputeKernelWorkGroupInfo.CompileWorkGroupSize, CL12.GetKernelWorkGroupInfo));
        }

        ///<summary>
        ///Gets the preferred multiple of workgroup size for launch.
        ///</summary>
        ///<param name = "device"> One of the <see cref = "ComputeKernel.Program.Device"/> s. </param>
        ///<returns> The preferred multiple of workgroup size for launch. </returns>
        ///<remarks> Specified a workgroup size returned by this query as the value of the local work size argument to Compute Command Queue. Exception will not fail to enqueue the kernel for execution unless   the work-group size specified is larger than the device maximum. </remarks>
        ///<remarks> Requires OpenCL 1.1. </remarks>
        public long GetPreferredWorkGroupSizeMultiple(ComputeDevice device)
        {
            return (long)GetInfo<CLKernelHandle, CLDeviceHandle, ComputeKernelWorkGroupInfo, IntPtr>(
                Handle, device.Handle, ComputeKernelWorkGroupInfo.PreferredWorkGroupSizeMultiple, CL12.GetKernelWorkGroupInfo);
        }

        ///<summary>
        ///Gets the minimum amount of memory, in bytes, used by each work-item in the kernel.
        ///</summary>
        ///<param name = "device"> One of the <see cref = "ComputeKernel.Program.Device"/> s. </param>
        ///<returns> The minimum amount of memory, in bytes, used by each work-item in the kernel. </returns>
        ///__ private ___ </c> or <c> private ___ ___ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _   </c> qualifier. </remarks>
        public long GetPrivateMemorySize(ComputeDevice device)
        {
            return GetInfo<CLKernelHandle, CLDeviceHandle, ComputeKernelWorkGroupInfo, long>(
                Handle, device.Handle, ComputeKernelWorkGroupInfo.PrivateMemorySize, CL12.GetKernelWorkGroupInfo);
        }

        ///<summary>
        ///<See cref = "ComputeKernel"/> on a <see cref = "ComputeDevice"/> Gets the maximum work - group size.
        ///</summary>
        ///<param name = "device"> One of the <see cref = "ComputeKernel.Program.Device"/> s. </param>
        ///<return> The maximum work-group size is can be used to execute the <see cref = "ComputeKernel"/> on <paramref name = "device"/>. </returns>
        public long GetWorkGroupSize(ComputeDevice device)
        {
            return (long)GetInfo<CLKernelHandle, CLDeviceHandle, ComputeKernelWorkGroupInfo, IntPtr>(
                    Handle, device.Handle, ComputeKernelWorkGroupInfo.WorkGroupSize, CL12.GetKernelWorkGroupInfo);
        }

        ///<summary>
        ///Sets an argument of the <see cref = "ComputeKernel"/> (no argument tracking).
        ///</summary>
        ///<param name = "index"> The argument index. </param>
        ///<param name = "dataSize"> The size of the argument data in bytes. </param>
        ///<param name = "dataAddr"> A pointer to the data that should be used as the argument value. </param>
        ///<remarks>
        ///Arguments to the kernel being referred by indices that go from 0 for the leftmost argument to n-1, where n is the total number of arguments declared by the kernel.
        ///2.540
        ///It is up to the user to reference the kernel arguments (the prevent them from being garbage collected) until the kernel has finished execution.
        ///</remarks>
        public void SetArgument(int index, IntPtr dataSize, IntPtr dataAddr)
        {
            ComputeErrorCode error = CL12.SetKernelArg(Handle, index, dataSize, dataAddr);
            ComputeException.ThrowOnError(error);
        }

        ///<summary>
        ///Sets the size in bytes of an argument specfied with the <c> local </c> or <c> __ local </c> address space qualifier.
        ///</summary>
        ///<param name = "index"> The argument index. </param>
        ///<param name = "dataSize"> The size of the argument data in bytes. </param>
        ///<remarks> Arguments to the kernel being referred by indices that go from 0 to the leftmost argument to n-1, where n is the total number of arguments declared by the kernel. </remarks>
        public void SetLocalArgument(int index, long dataSize)
        {
            SetArgument(index, new IntPtr(dataSize), IntPtr.Zero);
        }

        ///<summary>
        ///Sets a <c> T * </c>, <c> image 2 d - t </c> or <c> image 3 d - t </c> argument of the <see cref = "ComputeKernel"/>.
        ///</summary>
        ///<param name = "index"> The argument index. </param>
        ///<param name = "memObj"> The <see cref = "ComputeMemory"/> that is passed as the argument. </param>
        ///<remarks> This method will automatically track <paramref name = "memObj"/> to prevent it from being collected by the GC. Arguments to the kernel being referred by indices that go from 0 for the leftmost argument to n-   1, where n is the total number of arguments declared by the kernel. </Remarks>
        public void SetMemoryArgument(int index, ComputeMemory memObj)
        {
            SetValueArgument<CLMemoryHandle>(index, memObj.Handle);
        }

        ///<summary>
        ///Sets a <c> sampler_t </c> argument of the <see cref = "ComputeKernel"/>.
        ///</summary>
        ///<param name = "index"> The argument index. </param>
        ///<param name = "sampler"> The <see cref = "ComputeSampler"/> that is passed as the argument. </param>
        ///Arguments to the kernel being referred by indices that go from 0 for the leftmost argument to n-   1, where n is the total number of arguments declared by the kernel. </Remarks>
        public void SetSamplerArgument(int index, ComputeSampler sampler)
        {
            SetValueArgument<CLSamplerHandle>(index, sampler.Handle);
        }

        ///<summary>
        ///Sets a value argument of the <see cref = "ComputeKernel"/>.
        ///</summary>
        ///<typeparam name = "T"> The type of the argument. </typeparam>
        ///<param name = "index"> The argument index. </param>
        ///<param name = "data"> The data that is passed as the argument value. </param>
        ///<remarks> Arguments to the kernel being referred by indices that go from 0 to the leftmost argument to n-1, where n is the total number of arguments declared by the kernel. </remarks>
        public void SetValueArgument<T>(int index, T data) where T : struct
        {
            GCHandle gcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                SetArgument(
                    index,
                    new IntPtr(Marshal.SizeOf(typeof(T))),
                    gcHandle.AddrOfPinnedObject());
            }
            finally
            {
                gcHandle.Free();
            }
        }

        #endregion

        #region Protected methods

        ///<summary>
        ///Releases the associated OpenCL object.
        ///</summary>
        ///<param name = "manual"> Specifies the operation mode of this method. </param>
        ///<remarks> <paramref name = "manual"/> must be <c> true </c> if this method is invoked directly by the application. </remarks>
        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
                CL12.ReleaseKernel(Handle);
                Handle.Invalidate();
            }
        }

        #endregion
    }
}