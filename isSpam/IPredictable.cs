using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    internal interface IPredictable
    {
        public static string ? DataPath { get; set; }
        
        public static string ? UserInput { get; set; }

        public static string ? ModelPath { get; set; }


        public bool PredictSpam();

        public void Train();
    }
}