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
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using System.Threading;
    using Cloo.Bindings;

    ///<summary>
    ///Represents an OpenCL context.
    ///</summary>
    ///<remarks> The environment within which the kernels execute and the domain in which synchronization and memory management is defined. </remarks>
    ///2.540
    ///<example>
    ///This example shows how to create a <see cref = "ComputeContext"/> that is able to share data with an OpenGL context in a Microsoft Windows OS:
    ///<code>
    ///<! [CDATA [
    ///,
    ///// NOTE: If you see some some C # bits surrounding this code section, ignore them. They are not part of the code.
    ///,
    ///// We will need the device context, which is acquired through OS specific function.
    ///[DllImport ("opengl32.dll")]
    ///extern static IntPtr wglGetCurrentDC ();
    ///,
    ///// Query the device context.
    ///IntPtr deviceContextHandle = wglGetCurrentDC ();
    ///,
    ///// Select a platform which is capable of OpenCL / OpenGL interop.
    ///ComputePlatform platform = ComputePlatform.GetByName (name);
    ///,
    ///// Create the context property list and populate it.
    ///ComputeContextProperty p1 = new ComputeContextProperty (ComputeContextPropertyName.Platform, platform.Handle.Value);
    ///ComputeContextProperty p2 = new ComputeContextProperty (ComputeContextPropertyName.CL_GL_CONTEXT_KHR, openGLContextHandle);
    ///ComputeContextProperty p3 = new ComputeContextProperty (ComputeContextPropertyName.CL_WGL_HDC_KHR, deviceContextHandle);
    ///ComputeContextPropertyList cpl = new ComputeContextPropertyList (new ComputeContextProperty [] {p1, p2, p3});
    ///,
    ///// Create the context., You will want this on a GPU but other options might be available as well.
    ///ComputeContext context = new ComputeContext (ComputeDeviceTypes.Gpu, cpl, null, IntPtr.Zero);
    ///,
    ///// Create a shared OpenCL / OpenGL buffer.
    ///// The generic type should match the type of data that the buffer contains.
    ///// glBufferId is an existing OpenGL buffer identifier.
    ///ComputeBuffer <float> clglBuffer = ComputeBuffer.CreateFromGLBuffer <float> (context, ComputeMemoryFlags.ReadWrite, glBufferId);
    ///,
    ///]]>
    ///</code>
    ///Before working with the <c> clglBuffer </c> you should have surely two things: ¬
    ///1) OpenGL is not using <c> glBufferId </c>. You can achieve this by calling <c> glFinish </c>.
    ///2) Make it available to OpenCL through the <see cref = "ComputeCommandQueue.AcquireGLObjects"/> method.
    ///After that, call <see cref = "ComputeCommandQueue.ReleaseGLObjects"/> to make the buffer available to OpenGL again. When finished, you should wait until <cl> clglBuffer </c>
    ///</example>
    ///<seealso cref = "ComputeDevice"/>
    ///<seealso cref = "ComputePlatform"/>
    public class ComputeContext : ComputeResource
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<ComputeDevice> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputePlatform platform;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ComputeContextPropertyList properties;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeContextNotifier callback;

        #endregion

        #region Properties

        ///<summary>
        ///The handle of the <see cref = "ComputeContext"/>.
        ///</summary>
        public CLContextHandle Handle
        {
            get;
            protected set;
        }

        ///<summary>
        ///Gets a read-only collection of the <see cref = "ComputeDevice"/> s of the <see cref = "ComputeContext"/>.
        ///</summary>
        ///<value> A read-only collection of the <see cref = "ComputeDevice"/> s of the <see cref = "ComputeContext"/>. </value>
        public ReadOnlyCollection<ComputeDevice> Devices { get { return devices; } }

        ///<summary>
        ///Gets the <see cref = "ComputePlatform"/> of the <see cref = "ComputeContext"/>.
        ///</summary>
        ///<value> The <see cref = "ComputePlatform"/> of the <see cref = "ComputeContext"/>. </value>
        public ComputePlatform Platform { get { return platform; } }

        ///<summary>
        ///Gets a collection of <see cref = "ComputeContextProperty"/> s of the <see cref = "ComputeContext"/>.
        ///</summary>
        ///<value> A collection of <see cref = "ComputeContextProperty"/> s of the <see cref = "ComputeContext"/>. </value>
        public ComputeContextPropertyList Properties { get { return properties; } }

        #endregion

        #region Constructors

        ///<summary>
        ///Creates a new <see cref = "ComputeContext"/> on a collection of <see cref = "ComputeDevice"/> s.
        ///</summary>
        ///<param name = "devices"> A collection of <see cref = "ComputeDevice"/> s to associate with the <see cref = "ComputeContext"/>. </param>
        ///<param name = "properties"> A <see cref = "ComputeContextPropertyList"/> of the <see cref = "ComputeContext"/>. </param>
        ///<param name = "notify"> A delegate instance that relates to a notification routine. The OpenCL implementation to report information on errors that occur in the <see cref = "ComputeContext"/>   The callback function may be invoked asynchronously by the OpenCall function. The callback function is thread - safe and that delegate instance does not get collected by the Garbage Collector until <see cref = "ComputeContext"  /> is placed. If <paramref name = "notify"/> is <c> null </c>, no callback function is registered. </param>
        ///<param name = "notifyDataPtr"> Optional user data that will be passed to <paramref name = "notify"/>. </param>
        public ComputeContext(ICollection<ComputeDevice> devices, ComputeContextPropertyList properties, ComputeContextNotifier notify, IntPtr notifyDataPtr)
        {
            int handleCount;
            CLDeviceHandle[] deviceHandles = ComputeTools.ExtractHandles(devices, out handleCount);
            IntPtr[] propertyArray = (properties != null) ? properties.ToIntPtrArray() : null;
            callback = notify;

            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL12.CreateContext(propertyArray, handleCount, deviceHandles, notify, notifyDataPtr, out error);
            ComputeException.ThrowOnError(error);
            
            SetID(Handle.Value);
            
            this.properties = properties;
            ComputeContextProperty platformProperty = properties.GetByName(ComputeContextPropertyName.Platform);
            this.platform = ComputePlatform.GetByHandle(platformProperty.Value);
            this.devices = GetDevices();

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
        }

        ///<summary>
        ///Creating a new <see cref = "ComputeContext"/> on all the <see cref = "ComputeDevice"/> s that match the specified <see cref = "ComputeDeviceTypes"/>
        ///</summary>
        ///<para name = "deviceType"> A bit-field that identifies the type of <see cref = "ComputeDevice"/> to associate with the <see cref = "ComputeContext"/>. </param>
        ///<param name = "properties"> A <see cref = "ComputeContextPropertyList"/> of the <see cref = "ComputeContext"/>. </param>
        ///<param name = "notify"> A delegate instance that relates to a notification routine. The OpenCL implementation to report information on errors that occur in the <see cref = "ComputeContext"/>   The callback function may be invoked asynchronously by the OpenCall function. The callback function is thread - safe and that delegate instance does not get collected by the Garbage Collector until <see cref = "ComputeContext"  /> is placed. If <paramref name = "notify"/> is <c> null </c>, no callback function is registered. </param>
        ///<param name = "userDataPtr"> Optional user data that will be passed to <paramref name = "notify"/>. </param>
        public ComputeContext(ComputeDeviceTypes deviceType, ComputeContextPropertyList properties, ComputeContextNotifier notify, IntPtr userDataPtr)
        {
            IntPtr[] propertyArray = (properties != null) ? properties.ToIntPtrArray() : null;
            callback = notify;

            ComputeErrorCode error = ComputeErrorCode.Success;
            Handle = CL12.CreateContextFromType(propertyArray, deviceType, notify, userDataPtr, out error);
            ComputeException.ThrowOnError(error);

            SetID(Handle.Value);

            this.properties = properties;
            ComputeContextProperty platformProperty = properties.GetByName(ComputeContextPropertyName.Platform);
            this.platform = ComputePlatform.GetByHandle(platformProperty.Value);
            this.devices = GetDevices();

            Trace.WriteLine("Create " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
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
            if (manual)
            {
                //free managed resources
            }

            //free native resources
            if (Handle.IsValid)
            {
                Trace.WriteLine("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
                CL12.ReleaseContext(Handle);
                Handle.Invalidate();
            }
        }

        #endregion

        #region Private methods

        private ReadOnlyCollection<ComputeDevice> GetDevices()
        {
            List<CLDeviceHandle> deviceHandles = new List<CLDeviceHandle>(GetArrayInfo<CLContextHandle, ComputeContextInfo, CLDeviceHandle>(Handle, ComputeContextInfo.Devices, CL12.GetContextInfo));
            List<ComputeDevice> devices = new List<ComputeDevice>();
            foreach (ComputePlatform platform in ComputePlatform.Platforms)
            {
                foreach (ComputeDevice device in platform.Devices)
                    if (deviceHandles.Contains(device.Handle))
                        devices.Add(device);
            }
            return new ReadOnlyCollection<ComputeDevice>(devices);
        }

        #endregion
    }
}