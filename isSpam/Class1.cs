using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    internal abstract class Classifier
    {
        public bool Predict()
        {
            return false;
        }

        public void Train()
        { 
        
        }


    }

    class LogisticRegressionClassifier : Classifier
    { 
    
    }

    class NaiveBayesClassifier : Classifier
    { 
    
    }

    class SupportVectorMachine : Classifier
    { 
    
    }
}
