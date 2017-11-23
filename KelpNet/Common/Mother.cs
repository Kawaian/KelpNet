using System;

namespace KelpNet.Common
{
    //Random numbers
    //In C#, when multiple instances of Random are simultaneously instantiated, only similar values ​​are emitted
    //It is necessary to manage it collectively in one place
    public class Mother
    {
#if DEBUG
        //Fix seed when debugging
        public static Random Dice = new Random(128);
#else
        public static Random Dice = new Random();
#endif
        static double Alpha, Beta, BoxMuller1, BoxMuller2;
        static bool Flip;
        public static double Mu = 0;
        public static double Sigma = 1;

        //Obtain normal distributed random numbers with mean mu and standard deviation sigma.   According to the Box-Muller method.
        public static double RandomNormal()
        {
            if (!Flip)
            {
                Alpha = Dice.NextDouble();
                Beta = Dice.NextDouble() * Math.PI * 2;
                BoxMuller1 = Math.Sqrt(-2 * Math.Log(Alpha));
                BoxMuller2 = Math.Sin(Beta);
            }
            else
            {
                BoxMuller2 = Math.Cos(Beta);
            }

            Flip = !Flip;

            return Sigma * (BoxMuller1 * BoxMuller2) + Mu;
        }
    }
}
