using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class AlphaVector
    {
        private Dictionary<State, double> m_dValues;
        public Action Action { get; private set; }
        public double this[State s]
        {
            get
            {
                if (m_dValues.ContainsKey(s))
                    return m_dValues[s];
                return 0.0;
            }
            set
            {
                if (value != 0.0)
                    m_dValues[s] = value;
            }
        }
        public IEnumerable<KeyValuePair<State, double>> Values
        {
            get
            {
                foreach (KeyValuePair<State, double> p in m_dValues)
                    yield return p;
            }
        }
        public AlphaVector()
        {
            Action = null;
            m_dValues = new Dictionary<State, double>();
        }

        public AlphaVector(Action a)
        {
            Action = a;
            m_dValues = new Dictionary<State, double>();
        }
        public AlphaVector(AlphaVector av)
        {
            Action = av.Action;
            m_dValues = new Dictionary<State, double>(av.m_dValues);
        }
        public static AlphaVector operator +(AlphaVector av1, AlphaVector av2)
        {
            AlphaVector avNew = new AlphaVector(av1);
            foreach (KeyValuePair<State, double> p in av2.Values)
            {
                if (!avNew.m_dValues.ContainsKey(p.Key))
                    avNew.m_dValues[p.Key] = 0.0;
                avNew.m_dValues[p.Key] += p.Value;
            }
            return avNew;
        }
        public static AlphaVector operator *(AlphaVector av, double dScalar)
        {
            AlphaVector avNew = new AlphaVector(av.Action);
            foreach (KeyValuePair<State, double> p in av.Values)
            {
                avNew.m_dValues[p.Key] = p.Value * dScalar;
            }
            return avNew;
        }
        
        public void InitAlphaVector(IEnumerable<State> States)
        {
            foreach(State s in States)
            {
                m_dValues[s] = s.Reward(Action);
            }
        }

        public double InnerProduct(BeliefState bs)
        {
            double dSum = 0.0;
            foreach (KeyValuePair<State, double> p in m_dValues)
                dSum += p.Value * bs[p.Key];
            return dSum;
        }
        public override bool Equals(object obj)
        {
            if (obj is AlphaVector)
            {
                AlphaVector av = (AlphaVector)obj;
                foreach (KeyValuePair<State, double> p in m_dValues)
                {
                    if (Math.Abs(p.Value - av[p.Key]) > 0.001)
                        return false;
                }
                foreach (KeyValuePair<State, double> p in av.m_dValues)
                {
                    if (Math.Abs(p.Value - this[p.Key]) > 0.001)
                        return false;
                }
                return true;
            }
            return false;
        }
        public override int GetHashCode()
        {
            int iSum = 0;
            foreach (KeyValuePair<State, double> p in m_dValues)
                iSum += (int)(p.Value * 100);
            return iSum;
        }
    }
}
