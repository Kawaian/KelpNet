namespace KelpNetTester.Tests
{
    partial class Test13WinForm
    {
        ///<summary>
        ///It is a necessary designer variable.
        ///</summary>
        private System.ComponentModel.IContainer components = null;

        ///<summary>
        ///Clean up all resources in use.
        ///</summary>
        ///<param name = "disposing"> Specify true to destroy the managed resource, false otherwise.   </param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (this.components != null))
            {
                this.components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows フォーム デザイナーで生成されたコード

        ///<summary>
        ///It is a method necessary for designer support.   The contents of this method
        ///Please do not change with code editor.
        ///</summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.SuspendLayout();
            //,
            //timer1
            //,
            this.timer1.Enabled = true;
            this.timer1.Interval = 300;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            //,
            //Test 13
            //,
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.ClientSize = new System.Drawing.Size(284, 261);
            this.DoubleBuffered = true;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.Name = "Test13";
            this.Text = "Test13";
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Timer timer1;
    }
}

