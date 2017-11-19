using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Cloo;
using KelpNet.Properties;

namespace KelpNet.Common
{
    ///<summary>
    ///The types of devices.
    ///</summary>
    [Flags]
    public enum ComputeDeviceTypes : long
    {
        Default = 1 << 0,
        Cpu = 1 << 1,
        Gpu = 1 << 2,
        Accelerator = 1 << 3,
        All = 0xFFFFFFFF
    }

    //GPU Manager responsible for related processes
    public class Weaver
    {
        public const string USE_DOUBLE_HEADER_STRING =
@"
  #if __OPENCL__VERSION__ <= __CL_VERSION_ 1 _ 1
  #if defined (cl_khr_fp64)
  # pragma OPENCL EXTENSION cl_khr_fp 64: enable
  # elif defined (cl_amd_fp 64)
  # pragma OPENCL EXTENSION cl_amd_fp 64: enable
  # endif
  # endif
  ";

        public const string REAL_HEADER_STRING =
@"
  //! REAL is provided by compiler option
  typedef REAL Real;
  ";

        public static ComputeContext Context;
        private static ComputeDevice[] Devices;
        public static ComputeCommandQueue CommandQueue;
        public static int DeviceIndex;
        public static bool Enable;
        public static ComputePlatform Platform;
        public static Dictionary<string, string> KernelSources = new Dictionary<string, string>();

        public static string GetKernelSource(string functionName)
        {
            if (!KernelSources.ContainsKey(functionName))
            {
                byte[] binary = (byte[])Resources.ResourceManager.GetObject(functionName);
                if (binary == null) throw new Exception("Resource file acquisition failed \n Resource name:" + functionName);

                using (StreamReader reader = new StreamReader(new MemoryStream(binary)))
                {
                    KernelSources.Add(functionName, reader.ReadToEnd());
                }
            }

            return KernelSources[functionName];
        }

        public static void Initialize(ComputeDeviceTypes selectedComputeDeviceTypes, int platformId = 0, int deviceIndex = 0)
        {
            Platform = ComputePlatform.Platforms[platformId];

            Devices = Platform
                .Devices
                .Where(d => (long)d.Type == (long)selectedComputeDeviceTypes)
                .ToArray();

            DeviceIndex = deviceIndex;

            if (Devices.Length > 0)
            {
                Context = new ComputeContext(
                    Devices,
                    new ComputeContextPropertyList(Platform),
                    null,
                    IntPtr.Zero
                    );

                CommandQueue = new ComputeCommandQueue(
                    Context,
                    Devices[DeviceIndex],
                    ComputeCommandQueueFlags.None
                    );

                Enable = true;
            }
        }

        public static ComputeProgram CreateProgram(string source)
        {
            string realType = Marshal.SizeOf(typeof(Real)) == Marshal.SizeOf(typeof(double)) ? "double" : "float";

            //for precision setting of floating point
            source = REAL_HEADER_STRING + source;

            //Add at double precision
            if (realType == "double")
            {
                source = USE_DOUBLE_HEADER_STRING + source;
            }

            ComputeProgram program = new ComputeProgram(Context, source);

            try
            {
                program.Build(Devices, string.Format("-D REAL={0} -Werror", realType), null, IntPtr.Zero);
            }
            catch
            {
                MessageBox.Show(program.GetBuildLog(Devices[DeviceIndex]));
            }

            return program;
        }
    }
}
