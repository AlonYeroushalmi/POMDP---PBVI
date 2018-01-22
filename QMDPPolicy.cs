using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class QMDPPolicy : Policy
    {
        private MDPValueFunction m_vValueFunction;
        private Domain m_dDomain;
        public QMDPPolicy(MDPValueFunction v, Domain d)
        {
            m_vValueFunction = v;
            m_dDomain = d;
        }
        public override Action GetAction(BeliefState bs)
        {
            //your code here
            throw new NotImplementedException();
        }
    }
}
