using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace POMDP
{
    class MazeAction : Action
    {
        public string Name { get; private set; }
        public MazeAction(string sName)
        {
            Name = sName;
        }
        public override string ToString()
        {
            return Name;
        }
        public override int GetHashCode()
        {
            return Name.GetHashCode();
        }
        public override bool Equals(object obj)
        {
            if (obj is MazeAction)
            {
                return Name.Equals(((MazeAction)obj).Name);
            }
            return false;
        }
    }
}
