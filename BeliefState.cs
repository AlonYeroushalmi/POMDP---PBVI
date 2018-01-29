using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace POMDP
{
    class BeliefState
    {
        private Dictionary<State,double> m_dBeliefs; //each belief state is a vector containing the probability to be at each state
        private Domain m_dDomain;

        //returns the probability that this beliefstate is actually in state s
        public double this[State s]
        {
            get 
            { 
                if(m_dBeliefs.ContainsKey(s))    
                    return m_dBeliefs[s];
                return 0.0;
            }
            set { m_dBeliefs[s] = value; }
        }

        //returns a dict with all the states in this belief state which their probability is bigger than dMin
        public IEnumerable<KeyValuePair<State, double>> Beliefs(double dMin) 
        {
            foreach (KeyValuePair<State, double> p in m_dBeliefs)
                if (p.Value >= dMin)
                    yield return p;
        }

        //constructor
        public BeliefState(Domain d)
        {
            m_dDomain = d;
            m_dBeliefs = new Dictionary<State, double>();
        }

        public BeliefState(BeliefState bs)
        {
            this.m_dDomain = bs.m_dDomain;
            this.m_dBeliefs = new Dictionary<State, double>();
            foreach (KeyValuePair<State,double> p in bs.m_dBeliefs)
            {
                m_dBeliefs.Add(p.Key, p.Value);
            }
        }

        //add a state with its probability
        private void AddBelief(State s, double dProb)
        {
            if (!m_dBeliefs.ContainsKey(s))
                m_dBeliefs[s] = 0;
            m_dBeliefs[s] += dProb;
        }

        // t(b,a,b') = pr(b'| a,b) = (sum over all o in omega) pr(b'|a,o,b) * pr(o|a,b). lecture 13, page 3
        public BeliefState Next(Action a, Observation o)
        {
            BeliefState bsNext = new BeliefState(m_dDomain);
            //double sumOfBTag = 0;
            foreach (State sTag in m_dDomain.States)
            {
                double stateProbabilityInBtag;
                stateProbabilityInBtag = sTag.ObservationProbability(a, o) * transitionProbabilityForEachState(sTag, a) / probabilityOfObservationGivenAB(a, o);
                bsNext.AddBelief(sTag, stateProbabilityInBtag);
            }
            return bsNext;
            //foreach (State sTag in m_dDomain.States)
            //{
            //    //pr(b'|a,o,b)
            //    double currBTag = CalculateBTagForEachState(sTag, a, o);
            //    sumOfBTag += currBTag;
            //    bsNext.AddBelief(sTag, currBTag);
            //}
            //foreach (State sTag in m_dDomain.States)
            //{
            //    bsNext.m_dBeliefs[sTag] = bsNext.m_dBeliefs[sTag]/sumOfBTag;
            //}
            //Debug.Assert(bsNext.Validate());
            //return bsNext;
        }

        private double probabilityOfObservationGivenAB(Action a, Observation o)
        {
            double sTagSum = 0;
            foreach(State sTag in m_dDomain.States)
            {
                sTagSum += sTag.ObservationProbability(a, o) * transitionProbabilityForEachState(sTag, a);
            }
            return sTagSum;
        }

        private double transitionProbabilityForEachState(State sTag, Action a)
        {
            double sumS = 0;
            foreach(State s in m_dDomain.States)
            {
                sumS += s.TransitionProbability(a, sTag);
            }
            return sumS;
        }


        //calculates b'(s'|b,a,o)
        private double CalculateBTagForEachState(State sTag, Action a, Observation o)
        {
            double sumOfAllStags = sTag.SumTransitionProbabilityOfAllStates(a);
            return sTag.ObservationProbability(a, o) * sumOfAllStags * this[sTag];
        }

        public override string ToString()
        {
            string s = "<";
            foreach (KeyValuePair<State, double> p in m_dBeliefs)
            {
                if( p.Value > 0.01 )
                    s += p.Key + "=" + p.Value.ToString("F") + ",";
            }
            s += ">";
            return s;
        }

        public bool Validate()
        {
            //validate that every state appears at most once
            List<State> lStates = new List<State>(m_dBeliefs.Keys);
            int i = 0, j = 0;
            for (i = 0; i < lStates.Count; i++)
            {
                for (j = i + 1; j < lStates.Count; j++)
                {
                    if (lStates[i].Equals(lStates[j]))
                        return false;
                }
            }
            double dSum = 0.0;
            foreach (double d in m_dBeliefs.Values)
                dSum += d;
            if (Math.Abs(1.0 - dSum) > 0.001)
                return false;
            return true;
        }

        public State RandomState()
        {
            Random rand = new Random();
            List<State> SteteList = Enumerable.ToList(m_dBeliefs.Keys);
            int size = m_dBeliefs.Count;
            return SteteList[rand.Next(size)];
        }

        private double ObservationProbability(Observation o, Action a)
        {
            double oProb = 0.0;
            foreach(State sTag in m_dDomain.States)
            {
                foreach(State s in m_dDomain.States)
                {
                    oProb += sTag.ObservationProbability(a, o) * 
                        s.TransitionProbability(a,sTag) * m_dBeliefs[s];
                }
            }
            return oProb;
        }

        public Observation RandomObservation(Action a)
        {
            double dRnd = RandomGenerator.NextDouble();
            double dProb = 0.0;
            foreach (Observation o in m_dDomain.Observations)
            {
                dProb = ObservationProbability(o, a);
                dRnd -= dProb;
                if (dRnd <= 0)
                    return o;
            }
            return null;//bugbug
        }

        //iterate through all states in the current belief state and return the sum of the reward* probability of being in this state 
        public double Reward(Action a)
        {
            double dSum = 0.0;
            foreach (KeyValuePair<State, double> p in m_dBeliefs)
            {
                dSum += p.Value * p.Key.Reward(a);
            }
            return dSum;
        }
    }
}
