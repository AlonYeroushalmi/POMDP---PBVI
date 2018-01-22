namespace POMDP
{
    partial class MazeViewer
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.MazePictureBox = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.MazePictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // MazePictureBox
            // 
            this.MazePictureBox.Location = new System.Drawing.Point(12, 12);
            this.MazePictureBox.Name = "MazePictureBox";
            this.MazePictureBox.Size = new System.Drawing.Size(331, 314);
            this.MazePictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.MazePictureBox.TabIndex = 0;
            this.MazePictureBox.TabStop = false;
            this.MazePictureBox.Paint += new System.Windows.Forms.PaintEventHandler(this.TrackPictureBox_Paint);
            // 
            // MazeViewer
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ButtonShadow;
            this.ClientSize = new System.Drawing.Size(355, 338);
            this.Controls.Add(this.MazePictureBox);
            this.Name = "MazeViewer";
            this.Text = "MazeViewer";
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.RaceViewer_Paint);
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.RaceViewer_FormClosed);
            ((System.ComponentModel.ISupportInitialize)(this.MazePictureBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox MazePictureBox;

    }
}