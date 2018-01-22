using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace POMDP
{
    class MDPValueFunction
    {
        private Dictionary<State, Dictionary<Action, double>> m_dQValues; // 
        private Dictionary<State, double> m_dValues;  
        private Dictionary<State, Action> m_dBestActions; //contains the best current action for each state.
        private Domain m_dDomain;
        public double MaxValue { get; private set; }
        public double MinValue { get; private set; }

        public MDPValueFunction(Domain d)
        {
            m_dQValues = new Dictionary<State, Dictionary<Action, double>>();
            m_dValues = new Dictionary<State, double>();
            m_dBestActions = new Dictionary<State, Action>();
            m_dDomain = d;
            MaxValue = 0.0;
            MinValue = 0.0;
        }

        //returns the value of state s
        public double ValueAt(State s)
        {
            if( m_dValues.ContainsKey(s) )
                return m_dValues[s];
            return 0.0;
        }

        public double QValueAt(State s, Action a)
        {
            return m_dQValues[s][a];
        }

        //returns the best current action for state s
        public Action GetAction(State s)
        {
            if (m_dBestActions.ContainsKey(s))
                return m_dBestActions[s];
            return null;
        }

        private Dictionary<State,List<State>> Init(double dInitValue)
        {
            Dictionary<State, List<State>> dPreds = new Dictionary<State, List<State>>(); //predecessors for each state 
            foreach (State s in m_dDomain.States)
            {

                m_dValues[s] = dInitValue;
                m_dBestActions[s] = null;
                m_dQValues[s] = new Dictionary<Action, double>();
                foreach (Action a in m_dDomain.Actions)
                {
                    m_dQValues[s][a] = 0.0;
                    foreach (State sTag in s.Successors(a))
                    {
                        if (!dPreds.ContainsKey(sTag))
                            dPreds[sTag] = new List<State>();
                        if (!dPreds[sTag].Contains(s))
                            dPreds[sTag].Add(s);
                    }
                }
            }
            return dPreds;
        }

        private double ComputeQValue(State s, Action a, double dDefaultStateValue)
        {
            double dImmediateReward = s.Reward(a);
            double dFutureReward = 0.0, dTr = 0.0 ;
            foreach (State sTag in s.Successors(a))
            {
                dTr = s.TransitionProbability(a, sTag);
                if(m_dValues.ContainsKey(sTag))
                    dFutureReward += dTr * m_dValues[sTag];
                else
                    dFutureReward += dTr * dDefaultStateValue;
            }
            return dImmediateReward + dFutureReward * m_dDomain.DiscountFactor;
        }
        private double ComputeValue(State s, out Action aBest)
        {
            return ComputeValue(s, out aBest, 0.0);
        }

        private double ComputeValue(State s, out Action aBest, double dDefaultStateValue)
        {
            double dQValue = 0.0, dMaxQValue = double.NegativeInfinity ;
            aBest = null;
            foreach (Action a in m_dDomain.Actions)
            {
                dQValue = ComputeQValue(s, a, dDefaultStateValue);
                if (!m_dQValues.ContainsKey(s))
                    m_dQValues[s] = new Dictionary<Action, double>();
                m_dQValues[s][a] = dQValue;
                if (dQValue > dMaxQValue)
                {
                    dMaxQValue = dQValue;
                    aBest = a;
                }
            }
            return dMaxQValue;
        }
        public void ValueIteration(double dEpsilon)
        {
            int cUpdates = 0;
            TimeSpan tsExecutionTime;
            ValueIteration(dEpsilon, out cUpdates, out tsExecutionTime);
        }

        public void ValueIteration(double dEpsilon, out int cUpdates, out TimeSpan tsExecutionTime)
        {
            Debug.WriteLine("Starting value iteration");
            DateTime dtBefore = DateTime.Now;
            Init(0.0);
            int iIteration = 0;
            Action aBest = null;
            double dNewValue = 0.0, dDelta = 0.0, dMaxDelta = double.PositiveInfinity;
            double dMinValue = double.PositiveInfinity, dMaxValue = double.NegativeInfinity;
            cUpdates = 0;
            while (dMaxDelta > dEpsilon)
            {
                dMaxDelta = 0.0;
                dMinValue = double.PositiveInfinity;
                dMaxValue = double.NegativeInfinity;
                foreach (State sCurrent in m_dDomain.States)
                {
                    dNewValue = ComputeValue(sCurrent, out aBest);
                    m_dBestActions[sCurrent] = aBest;
                    dDelta = Math.Abs(dNewValue - m_dValues[sCurrent]);
                    if (dDelta > dMaxDelta)
                        dMaxDelta = dDelta;
                    if (dNewValue > dMaxValue)
                        dMaxValue = dNewValue;
                    if (dNewValue < dMinValue)
                        dMinValue = dNewValue;
                    m_dValues[sCurrent] = dNewValue;
                    cUpdates++;
                }
                MinValue = dMinValue;
                MaxValue = dMaxValue;
                iIteration++;
                Debug.Write("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b" + iIteration + " " + dMaxDelta);
            }
            tsExecutionTime = DateTime.Now - dtBefore;
            Debug.WriteLine("\nFinished value iteration");
        }
    }
}
