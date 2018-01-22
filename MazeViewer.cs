using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Threading;

namespace POMDP
{
    public partial class MazeViewer : Form
    {
        private MazeDomain m_mdMaze;
        internal MazeState CurrentState;
        internal MazeState CurretnUpdateState;
        internal BeliefState CurrentBelief;
        internal MazeObservation CurrentObservation;
        private int SCALE = 50;
        internal delegate void RefreshDelegate();
        internal RefreshDelegate RefreshForm;
        internal delegate void HideDelegate();
        internal HideDelegate HideForm;
        private bool m_bThreadRunning;
        public bool Active { get; private set; }

        internal MazeViewer(MazeDomain maze)
        {
            InitializeComponent();
            m_mdMaze = maze;
            Size = new Size(m_mdMaze.Width * SCALE + 50, 100 + (m_mdMaze.Height + 3) * SCALE);
            MazePictureBox.Size = new Size(m_mdMaze.Width * SCALE, ( m_mdMaze.Height + 3 ) * SCALE);
            RefreshForm = new RefreshDelegate(RefreshFormMethod);
            HideForm = new HideDelegate(HideFormMethod);
            CurrentState = null;
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.SupportsTransparentBackColor | ControlStyles.AllPaintingInWmPaint, true);
            Active = false;
            m_bThreadRunning = false;
            CurretnUpdateState = null;
        }

        private void RaceViewer_Paint(object sender, PaintEventArgs e)
        {
        }

        private void DrawTrial(Graphics g)
        {
            Pen pThin = new Pen(Color.Black, 2);
            Pen pThick = new Pen(Color.Black, 5);
            int iX = 0, iY = 0;
            int iStartX = 0, iStartY = 0;
            int iEndX = iStartX, iEndY = iStartY;

            //g.FillRegion(Brushes.Gray, MazePictureBox.Region);

            for (iX = 0; iX < m_mdMaze.Width; iX++)
            {
                for (iY = 0; iY < m_mdMaze.Height; iY++)
                {
                    if (!m_mdMaze.BlockedSquare(iX, iY))
                    {
                        if (m_mdMaze.IsTargetSqaure(iX, iY))
                            g.FillRectangle(Brushes.Green, iX * SCALE, iY * SCALE, SCALE, SCALE);
                        else
                            g.FillRectangle(Brushes.White, iX * SCALE, iY * SCALE, SCALE, SCALE);
                        g.DrawRectangle(pThin, iX * SCALE, iY * SCALE, SCALE, SCALE);
                    }
                }
            }
            g.FillEllipse(Brushes.Yellow, CurrentState.X * SCALE + 2, CurrentState.Y * SCALE + 2, SCALE - 4, SCALE - 4);

            foreach (KeyValuePair<State, double> pair in CurrentBelief.Beliefs(0.01))
            {
                MazeState ms = (MazeState)pair.Key;
                Pen p1 = new Pen(Color.Blue, (float)Math.Max(1, 20 * pair.Value));
                iStartX = (int)((ms.X + 0.5) * SCALE);
                iStartY = (int)((ms.Y + 0.5) * SCALE);
                iEndX = iStartX;
                iEndY = iStartY;
                if (ms.CurrentDirection == MazeState.Direction.North)
                    iEndY -= SCALE / 2;
                if (ms.CurrentDirection == MazeState.Direction.South)
                    iEndY += SCALE / 2;
                if (ms.CurrentDirection == MazeState.Direction.East)
                    iEndX += SCALE / 2;
                if (ms.CurrentDirection == MazeState.Direction.West)
                    iEndX -= SCALE / 2;
                g.DrawLine(p1, iStartX, iStartY, iEndX, iEndY);
            }
            if (CurrentState != null)
            {
                iStartX = (int)((CurrentState.X + 0.5) * SCALE);
                iStartY = (int)((CurrentState.Y + 0.5) * SCALE);
                iEndX = iStartX;
                iEndY = iStartY;
                if (CurrentState.CurrentDirection == MazeState.Direction.North)
                    iEndY -= SCALE / 2;
                if (CurrentState.CurrentDirection == MazeState.Direction.South)
                    iEndY += SCALE / 2;
                if (CurrentState.CurrentDirection == MazeState.Direction.East)
                    iEndX += SCALE / 2;
                if (CurrentState.CurrentDirection == MazeState.Direction.West)
                    iEndX -= SCALE / 2;
                g.DrawLine(pThick, iStartX, iStartY, iEndX, iEndY);

            }
            Text = CurrentState.ToString();
            if (CurrentObservation.FrontWall)
            {
                iStartX = 0;
                iStartY = (m_mdMaze.Height + 1) * SCALE;
                iEndX = SCALE;
                iEndY = iStartY;
                g.DrawLine(pThick, iStartX, iStartY, iEndX, iEndY);
            }
            if (CurrentObservation.RightWall)
            {
                iStartX = SCALE;
                iStartY = (m_mdMaze.Height + 1) * SCALE;
                iEndX = iStartX;
                iEndY = iStartY + SCALE;
                g.DrawLine(pThick, iStartX, iStartY, iEndX, iEndY);
            }
            if (CurrentObservation.BackWall)
            {
                iStartX = 0;
                iStartY = (m_mdMaze.Height + 2) * SCALE;
                iEndX = SCALE;
                iEndY = iStartY;
                g.DrawLine(pThick, iStartX, iStartY, iEndX, iEndY);
            }
            if (CurrentObservation.LeftWall)
            {
                iStartX = 0;
                iStartY = (m_mdMaze.Height + 1) * SCALE;
                iEndX = 0;
                iEndY = iStartY + SCALE;
                g.DrawLine(pThick, iStartX, iStartY, iEndX, iEndY);
            }
        }

        internal void RefreshFormMethod()
        {
            //Update();
            Refresh();
            Application.DoEvents();

        }

        public bool Start()
        {
            if (m_bThreadRunning)
                return false;
            m_bThreadRunning = true;
            Active = true;
            Thread t = new Thread(Run);
            t.Start();
            return true;
        }

        private void Run()
        {
            //ShowDialog();
            Show();
            while (Active)
            {
                MazePictureBox.Invalidate();
                Application.DoEvents();
                Thread.Sleep(10);
            }
        }
        internal void HideFormMethod()
        {
            Visible = false;
            Hide();
        }

        private void TrackPictureBox_Paint(object sender, PaintEventArgs e)
        {
            try
            {
                Graphics g = e.Graphics;
                DrawTrial(g);
            }
            catch (Exception ex)
            {
            }
        }

        private void RaceViewer_FormClosed(object sender, FormClosedEventArgs e)
        {
            Active = false;
        }

    }
}
