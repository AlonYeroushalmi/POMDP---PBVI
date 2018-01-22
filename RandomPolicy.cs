using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class RandomPolicy : Policy
    {
        private List<Action> m_lActions;
        public RandomPolicy(Domain d)
        {
            m_lActions = new List<Action>();
            foreach (Action a in d.Actions)
            {
                m_lActions.Add(a);
            }
        }


        public override Action GetAction(BeliefState bs)
        {
            int idx = RandomGenerator.Next(m_lActions.Count);
            return m_lActions[idx];
        }
    }
}
