using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace POMDP
{
    class RandomGenerator
    {
        private static Random m_rnd = new Random(11);
        public static void Init( int iSeed )
        {
            m_rnd = new Random(iSeed);
        }

        public static int Next( int iMax )
        {
            return m_rnd.Next(iMax);
        }
        public static double NextDouble()
        {
            double dRnd = m_rnd.NextDouble();
            //Debug.WriteLine("Rnd " + dRnd);
            return dRnd;
        }
    }
}
