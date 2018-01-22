using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    abstract class Policy
    {
        public abstract Action GetAction(BeliefState bs);
    }
}
