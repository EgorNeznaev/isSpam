using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text;

namespace isSpam
{
    internal class Program
    {
        //убрал нижние подчеркивания у полей поскольку они не приватные
        public static readonly string dataPath = Path.Combine("..", "..", "..", "data", "data_one.csv");
        public static readonly string userInputPath = Path.Combine("..", "..", "..", "User", "user.txt");
        public static readonly string modelPath = Path.Combine("..", "..", "..", "data", "Model.zip");
        
        static void Main(string[] args)
        {
            string choice = IO.Menu();
            bool prediction = default;

            switch (choice)
            {
                case "1":
                    var LGC = new LogisticRegressionClassifier();  
                    LGC.TrainSpam(dataPath, modelPath);
                    prediction = LGC.PredictSpam(userInputPath, modelPath);
                    break;
                case "2":
                    var NBC = new NaiveBayesClassifier();
                    prediction = NBC.PredictSpam(userInputPath, dataPath);
                    break;
                case "3":
                    MLContext mlContext = new MLContext();
                    (double[][] inputs, int[] labels) = SupportVectorMachine.PreprocessTextData(dataPath, mlContext);
                    var SVM = new SupportVectorMachine(inputs, labels);
                    prediction = SVM.SvmTraining(userInputPath, dataPath);
                    break;
                default:
                    Main(args);
                    break;
            }

            IO.PrintResult(prediction);
        }
    }
}