using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    public class PredictionsModel
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
