using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    public class IsSpamModel
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool IsSpam { get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
