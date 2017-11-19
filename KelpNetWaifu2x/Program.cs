using System;
using System.Windows.Forms;

namespace KelpNetWaifu2x
{
    static class Program
    {
        ///<summary>
        ///It is the main entry point of the application.
        ///</summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new FormMain());
        }
    }
}
