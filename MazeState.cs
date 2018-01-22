using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class MazeState : State
    {
        public const double ActionSuccessProbability = 0.9;

        public int X{ get; private set;}
        public int Y{ get; private set;}
        public enum Direction { North, East, South, West };
        public Direction CurrentDirection { get; private set; }
        private MazeDomain m_mdMaze;

        public MazeState(int iX, int iY, Direction d, MazeDomain md)
        {
            X = iX;
            Y = iY;
            CurrentDirection = d;
            m_mdMaze = md;
        }

        public Direction TurnLeft(Direction d)
        {
            if (d == Direction.North)
                return Direction.West;
            if (d == Direction.South)
                return Direction.East;
            if (d == Direction.West)
                return Direction.South;
            if (d == Direction.East)
                return Direction.North;
            return Direction.North;//bugbug
        }

        public Direction TurnRight(Direction d)
        {
            if (d == Direction.North)
                return Direction.East;
            if (d == Direction.South)
                return Direction.West;
            if (d == Direction.West)
                return Direction.North;
            if (d == Direction.East)
                return Direction.South;
            return Direction.North;//bugbug
        }
        private void Forward(int iStartX, int iStartY, out int iEndX, out int iEndY)
        {
            iEndX = iStartX;
            iEndY = iStartY;
            if (CurrentDirection == Direction.South && !m_mdMaze.BlockedSquare(iStartX, iStartY + 1))
                iEndY++;
            if (CurrentDirection == Direction.North && !m_mdMaze.BlockedSquare(iStartX, iStartY - 1))
                iEndY--;
            if (CurrentDirection == Direction.East && !m_mdMaze.BlockedSquare(iStartX + 1, iStartY))
                iEndX++;
            if (CurrentDirection == Direction.West && !m_mdMaze.BlockedSquare(iStartX - 1, iStartY))
                iEndX--;
        }

        public override IEnumerable<State> Successors(Action a)
        {
            MazeAction ma = (MazeAction)a;
            if (m_mdMaze.IsTargetSqaure(this) && ma.Name == "Forward")
            {
                yield return new MazeState(-1, -1, Direction.North, m_mdMaze);
            }
            else
            {
                yield return this;
                MazeState sTag = Apply(ma, true);
                if (sTag != this)
                    yield return sTag;
            }
        }

        private MazeState Apply(MazeAction ma, bool bSuccess)
        {
            if (m_mdMaze.IsTargetSqaure(this) && ma.Name == "Forward")
            {
                return new MazeState(-1, -1, Direction.North, m_mdMaze);
            }
            if (X == -1 && Y == -1)
                return this;
            if (!bSuccess)
                return this;
            if (ma.Name == "TurnLeft")
            {
                return new MazeState(X, Y, TurnLeft(CurrentDirection), m_mdMaze);
            }
            if (ma.Name == "TurnRight")
            {
                return new MazeState(X, Y, TurnRight(CurrentDirection), m_mdMaze);
            }
            if (ma.Name == "Forward")
            {
                int iEndX = 0, iEndY = 0;
                Forward(X, Y, out iEndX, out iEndY);
                if (iEndX != X || iEndY != Y)
                    return new MazeState(iEndX, iEndY, CurrentDirection, m_mdMaze);          
            }
            return this;
        }

        public override State Apply(Action a)
        {
            MazeAction ma = (MazeAction)a;
            if (RandomGenerator.NextDouble() < ActionSuccessProbability)
                return Apply(ma, true);
            else
                return Apply(ma, false);
        }

        public override double TransitionProbability(Action a, State sTag)
        {
            MazeAction ma = (MazeAction)a;
            if (m_mdMaze.IsTargetSqaure(this) && ma.Name == "Forward")
            {
                if (m_mdMaze.IsGoalState(sTag))
                    return 1.0;
                return 0.0;
            }

            double dProb = 0.0;
            if (sTag.Equals(this))
            {
                dProb += 1 - ActionSuccessProbability;
            }
            MazeState s = Apply(ma, true);
            if (s.Equals(sTag))
            {
                dProb += ActionSuccessProbability;
            }
            return dProb;
        }

        //Returns summary of the TransitionFunction for each state sTag over the current "this" state 
        public override double SumTransitionProbabilityOfAllStates(Action a)
        {
            double sum = 0;
            foreach(State s in m_mdMaze.States)
            {
                sum += TransitionProbability(a, s);
            }
            return sum;
        }

        public override double ObservationProbability(Action a, Observation o)
        {
            MazeObservation oTruth = m_mdMaze.GetWallConfiguration(this);
            MazeObservation mo = (MazeObservation)o;
            return mo.ProbabilitySame(oTruth);
        }

        public override double Reward(Action a)
        {
            if (m_mdMaze.IsTargetSqaure(this) && ((MazeAction)a).Name == "Forward")
                return m_mdMaze.MaxReward;
            return 0.0;
        }

        public override bool Equals(object obj)
        {
            MazeState s = (MazeState)obj;
            return s.X == X && s.Y == Y && s.CurrentDirection == CurrentDirection;
        }
        public override string ToString()
        {
            return "[" + X + "," + Y + "," + CurrentDirection + "]";
        }
        public override int GetHashCode()
        {
            return X * Y;
        }

        public override Observation RandomObservation(Action a)
        {
            double dRnd = RandomGenerator.NextDouble();
            double dProb = 0.0;
            foreach (Observation o in m_mdMaze.Observations)
            {
                dProb = ObservationProbability(a, o);
                dRnd -= dProb;
                if (dRnd <= 0)
                    return o;
            }
            return null;//bugbug
        }
    }
}
