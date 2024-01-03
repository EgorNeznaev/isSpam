using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    internal class TransformText
    {
        [ColumnName("Label")]
        public bool IsSpam { get; set; }
        public float[] Features { get; set; }
    }
}
