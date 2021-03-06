using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace KelpNet.Common.Tools
{
    //I am referring to here
    //http: d.hatena.ne.jp/tekk/20100131/1264913887

    public static class DeepCopyHelper
    {
        public static T DeepCopy<T>(T target)
        {
            T result;

            using (MemoryStream mem = new MemoryStream())
            {
                BinaryFormatter bf = new BinaryFormatter();

                try
                {
                    bf.Serialize(mem, target);
                    mem.Position = 0;
                    result = (T) bf.Deserialize(mem);
                }
                finally
                {
                    mem.Close();
                }
            }

            return result;
        }
    }
}
