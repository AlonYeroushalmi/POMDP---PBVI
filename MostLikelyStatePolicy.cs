using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class MostLikelyStatePolicy : Policy
    {
        private MDPValueFunction m_vValueFunction;
        public MostLikelyStatePolicy(MDPValueFunction v)
        {
            m_vValueFunction = v;
        }
        public override Action GetAction(BeliefState bs)
        {
            //your code here
            throw new NotImplementedException();
        }
    }
}
