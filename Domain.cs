using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace POMDP
{
    abstract class Domain
    {
        public abstract IEnumerable<State> States { get; }
        public abstract IEnumerable<Action> Actions { get; }
        public abstract IEnumerable<Observation> Observations { get; }
        public abstract BeliefState InitialBelief { get; }
        public abstract double MaxReward { get; }
        public abstract bool IsGoalState(State s);
        public abstract State GetState(int iStateIdx);
        public double DiscountFactor { get; protected set; }

        /* simulates your policy for a number of iterations multiple times, and computes the average reward obtained.
        To generate a single trial:
             1. Sample a starting state s from the initial belief state. 
             2. Repeat until goal is reached
                  a) compute the action a for the belief state.
                  b) sample the result of applying a to s, obtaining s'.
                  c) sample an observation o based on a and s' 
                  d) compute the new belief state given your old belief state, a, and o.
                  e) accumulate the reward
        cStepsPerTrial = Number of iterations 
        cTrials = number of times of itrating cStepsPerTrial times. */
        public double ComputeAverageDiscountedReward(Policy p, int cTrials, int cStepsPerTrial)
        {
            double accumulatedReward = 0;
            for ( int i=1; i<= cTrials; i++ )
            {
                int remainingSteps = cStepsPerTrial;
                BeliefState bs = InitialBelief;
                // step 1: Sample a starting state s from the initial belief state.
                State s = bs.RandomState();
                // step 2: Repeat until goal is reached
                while (!IsGoalState(s) && remainingSteps > 0)
                {
                    //step 2a: compute the action a for the belief state.
                    Action a = p.GetAction(bs);
                    //step 2b: sample the result of applying a to s, obtaining s'.
                    State sTag =  s.Apply(a);
                    double reward = bs.Reward(a);
                    // step 2c: sample an observation o based on a and s(implemented with RandomObservation?)
                    Observation o = s.RandomObservation(a);
                    //step 2d: compute the new belief state given your old belief state, a, and o.
                    BeliefState newBeliefState = bs.Next(a, o);
                    bs = newBeliefState; //change bs for next iteration
                    //step 2e: accumulate the reward
                    accumulatedReward += reward;
                    s = sTag;
                    remainingSteps--;
                }
            }
            return (accumulatedReward / cTrials);
        }

        public Action GetRandomAction(Random rand)
        {
            int r = rand.Next(Actions.Count());
            return Actions.ElementAt(r);
        }
    }
}

