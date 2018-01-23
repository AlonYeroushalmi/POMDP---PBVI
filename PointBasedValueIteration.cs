using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace POMDP
{
    class PointBasedValueIteration : Policy
    {
        private Domain m_dDomain;
        private List<AlphaVector> m_lVectors;// this is our V;
        private Dictionary<AlphaVector, Dictionary<Action, Dictionary<Observation, AlphaVector>>> m_dGCache;

        //constructor
        public PointBasedValueIteration(Domain d)
        {
            m_dDomain = d;
        }


        //problem: at the end of getAction, avBest is still null, maybe m_lVectors is null?
        public override Action GetAction(BeliefState bs)
        {
            AlphaVector avBest = null;
            ValueOf(bs, m_lVectors, out avBest); 
            return avBest.Action;
        }


        private AlphaVector G(Action a, Observation o, AlphaVector av)
        {
            if (!m_dGCache.ContainsKey(av))
                m_dGCache[av] = new Dictionary<Action, Dictionary<Observation, AlphaVector>>();
            if (!m_dGCache[av].ContainsKey(a))
                m_dGCache[av][a] = new Dictionary<Observation, AlphaVector>();
            if (m_dGCache[av][a].ContainsKey(o))
                return m_dGCache[av][a][o];
            AlphaVector avNew = new AlphaVector(a);
            foreach (State s in m_dDomain.States)
            {
                double dSum = 0.0;
                foreach (State sTag in m_dDomain.States)
                {
                    dSum += sTag.ObservationProbability(a, o) * s.TransitionProbability(a, sTag) * av[sTag];

                }
                avNew[s] = dSum;
            }
            m_dGCache[av][a][o] = avNew;
            return avNew;
        }

        //generate a new alpha vector from a belief state and an action
        private AlphaVector G(BeliefState bs, Action a)
        {
            AlphaVector avSum = new AlphaVector(a);
            AlphaVector avGMax = null;
            double dValue = 0.0, dMaxValue = double.NegativeInfinity;
            foreach (Observation o in m_dDomain.Observations)
            {
                dMaxValue = double.NegativeInfinity;
                avGMax = null;
                foreach (AlphaVector avCurrent in m_lVectors)
                {
                    AlphaVector avG = G(a, o, avCurrent);
                    dValue = avG.InnerProduct(bs);
                    if (dValue > dMaxValue)
                    {
                        dMaxValue = dValue;
                        avGMax = avG;
                    }
                }
                avSum += avGMax;
            }
            avSum *= m_dDomain.DiscountFactor;
            AlphaVector avResult = new AlphaVector(a);
            foreach (State s in m_dDomain.States)
            {
                avResult[s] = avSum[s] + s.Reward(a);
            }
            return avResult;
        }
        private AlphaVector Backup(BeliefState bs)
        {
            AlphaVector avBest = null;
            //AlphaVector avCurrent = null;
            double dMaxValue = double.NegativeInfinity, dValue = 0.0;
            foreach(Action aCurr in m_dDomain.Actions)
            {
                foreach(AlphaVector avCurr in m_lVectors)
                {
                    AlphaVector avBA = G(bs, aCurr);
                    dValue = avBA.InnerProduct(bs); // check this
                    if(dMaxValue < dValue)
                    {
                        dMaxValue = dValue;
                        avBest = avCurr;
                    }
                }
            }
            return avBest;
        }

        private List<BeliefState> SimulateTrial(Policy p, int cMaxSteps)
        {
            BeliefState bsCurrent = m_dDomain.InitialBelief, bsNext = null;
            State sCurrent = bsCurrent.RandomState(), sNext = null;
            Action a = null;
            Observation o = null;
            List<BeliefState> lBeliefs = new List<BeliefState>();
            while (!m_dDomain.IsGoalState(sCurrent) && lBeliefs.Count < cMaxSteps)
            {
                a = p.GetAction(bsCurrent);
                sNext = sCurrent.Apply(a);
                o = sNext.RandomObservation(a);
                bsNext = bsCurrent.Next(a, o);
                bsCurrent = bsNext;
                lBeliefs.Add(bsCurrent);
                sCurrent = sNext;
            }
            return lBeliefs;
        }
        private List<BeliefState> CollectBeliefs(int cBeliefs)
        {
            Debug.WriteLine("Started collecting " + cBeliefs + " points");
            RandomPolicy p = new RandomPolicy(m_dDomain);
            int cTrials = 100, cBeliefsPerTrial = cBeliefs / cTrials;
            List<BeliefState> lBeliefs = new List<BeliefState>();
            while (lBeliefs.Count < cBeliefs)
            {
                lBeliefs.AddRange(SimulateTrial(p, cBeliefsPerTrial));
            }
            Debug.WriteLine("Collected " + lBeliefs.Count + " points");
            return lBeliefs;
        }

        //problem: we send null lVectors, causes dMaxValue to be NegativeInfinity;
        private double ValueOf(BeliefState bs, List<AlphaVector> lVectors, out AlphaVector avBest)
        {
            double dValue = 0.0, dMaxValue = double.NegativeInfinity;
            avBest = null;
            foreach (AlphaVector av in lVectors)
            {
                dValue = av.InnerProduct(bs);
                
                if (dValue > dMaxValue)
                {
                    dMaxValue = dValue;
                    avBest = av;
                }
            }
            return dMaxValue;
        }
        private List<BeliefState> GenerateB(int cBeliefs, Random rand)
        {
            List<BeliefState> B = new List<BeliefState>();
            BeliefState initB = m_dDomain.InitialBelief;
            int n = cBeliefs;
            while(n > 0)
            {
                Action a = (m_dDomain.GetRandomAction(rand));
                Observation oCurr = initB.RandomObservation(a);
                BeliefState bNext = initB.Next(a, oCurr);
                B.Add(bNext);
                initB = bNext;
                n--;
            }
            return B;
        }

        private void InitV()
        {
            m_lVectors = new List<AlphaVector>();
            foreach(Action a in m_dDomain.Actions)
            {
                AlphaVector newAV = new AlphaVector(a);
                newAV.InitAlphaVector(m_dDomain.States);
                m_lVectors.Add(newAV);
            }
        }

        private BeliefState RandomBeliefState(List<BeliefState> B, Random rand)
        {
            int r = rand.Next(B.Count);
            return B[r];
        }

        public void PointBasedVI(int cBeliefs, int cMaxIterations)
        {
            Random rand = new Random();
            List<BeliefState> B = GenerateB(cBeliefs, rand);
            InitV();
            m_dGCache = new Dictionary<AlphaVector, Dictionary<Action, Dictionary<Observation, AlphaVector>>>();
            List<BeliefState> BTag;
            while (cMaxIterations > 0)
            {
                BTag = GenerateB(cBeliefs, rand);
                List<AlphaVector> VTag = new List<AlphaVector>();
                while(BTag.Count != 0)
                {
                    //choose arbitrary point in BTag to improve
                    BeliefState bCurr = RandomBeliefState(BTag, rand);
                    AlphaVector newAV = Backup(bCurr);
                    AlphaVector avBest;
                    double currValue = ValueOf(bCurr, m_lVectors, out avBest);
                    if (newAV.InnerProduct(bCurr) >= currValue)
                    {
                        //remove from B points whose value was improved by new newAV
                        BTag.Where( b => newAV.InnerProduct(b) >= ValueOf(b, m_lVectors, out AlphaVector avTmp)).ToList();
                        avBest = newAV;
                    }
                    else
                    {
                        BTag.Remove(bCurr);
                        avBest = ArgMax(m_lVectors, b: bCurr);
                    }
                    VTag.Add(avBest);
                }
                m_lVectors = VTag;
                cMaxIterations--;
            }
            
        }

        private AlphaVector ArgMax(List<AlphaVector> m_lVectors, BeliefState b)
        {
            AlphaVector maxAlphaVector = new AlphaVector();
            foreach(AlphaVector aVector in m_lVectors)
            {
                if (aVector.InnerProduct(b) > maxAlphaVector.InnerProduct(b))
                    maxAlphaVector = aVector;
            }
            return maxAlphaVector;
        }

    }
}
