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
        public static readonly string dataPath = Path.Combine("..", "..", "..", "data", "data_one.csv");
        public static readonly string userInputPath = Path.Combine("..", "..", "..", "User", "user.txt");
        public static readonly string modelPath = Path.Combine("..", "..", "..", "data", "Model.zip");

        static void Main(string[] args)
        {
            string userInput = File.ReadAllText(userInputPath, Encoding.UTF8);
            string choice = IO.Menu();
            bool prediction = default;

            AbstractClassifier obj = null;

            switch (choice)
            {
                case "1":
                    obj = new LogisticRegressionClassifier();                                        
                    break;
                case "2":
                    obj = new NaiveBayesClassifier();
                    break;
                case "3":
                    MLContext mlContext = new MLContext();
                    (double[][] inputs, int[] labels) = SupportVectorMachine.PreprocessTextData(dataPath, mlContext);
                    obj = new SupportVectorMachine(inputs, labels);
                    break;
                default:
                    Main(args);
                    break;
            }

            prediction = obj.PredictSpam(userInput);
            IO.PrintResult(prediction, userInput);
            Console.ReadLine();
            Main(args);
        }
    }
}