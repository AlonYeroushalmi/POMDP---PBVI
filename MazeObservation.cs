using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class MazeObservation : Observation
    {
        public const double WALL_DETECTION_ACCURACY = 0.9;
        public bool FrontWall{ get; private set;}
        public bool LeftWall{ get; private set;}
        public bool RightWall{ get; private set;}
        public bool BackWall { get; private set; }
        public MazeObservation(int iObservationIndex)
        {
            LeftWall = iObservationIndex % 2 == 1;
            iObservationIndex /= 2;

            BackWall = iObservationIndex % 2 == 1;
            iObservationIndex /= 2;

            RightWall = iObservationIndex % 2 == 1;
            iObservationIndex /= 2;

            FrontWall = iObservationIndex % 2 == 1;
            iObservationIndex /= 2;
        }

        public double ProbabilitySame(MazeObservation oTruth)
        {
            double dProb = 1.0;
            if (FrontWall == oTruth.FrontWall)
                dProb *= WALL_DETECTION_ACCURACY;
            else
                dProb *= (1 - WALL_DETECTION_ACCURACY);
            if (RightWall == oTruth.RightWall)
                dProb *= WALL_DETECTION_ACCURACY;
            else
                dProb *= (1 - WALL_DETECTION_ACCURACY);
            if (LeftWall == oTruth.LeftWall)
                dProb *= WALL_DETECTION_ACCURACY;
            else
                dProb *= (1 - WALL_DETECTION_ACCURACY);
            if (BackWall == oTruth.BackWall)
                dProb *= WALL_DETECTION_ACCURACY;
            else
                dProb *= (1 - WALL_DETECTION_ACCURACY);
            return dProb;
        }
    }
}
