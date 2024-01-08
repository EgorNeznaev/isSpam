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

            var LRC = new LogisticRegressionClassifier(dataPath, userInput, modelPath);
            var NBC = new NaiveBayesClassifier(dataPath, userInput, modelPath);
            var SVM = new SupportVectorMachine(dataPath, userInput, modelPath);

            var LRCT = new Thread(LRC.Train);
            var NBCT = new Thread(NBC.Train);
            var SVMT = new Thread(SVM.Train);

            LRCT.Start();
            NBCT.Start();
            SVMT.Start();

        sorryforthislabel:
            string choice = IO.Menu(userInput);
            bool prediction = default;

            switch (choice)
            {
                case "1":
                    LRCT.Join();
                    prediction = LRC.PredictSpam();
                    break;

                case "2":
                    NBCT.Join();
                    prediction = NBC.PredictSpam();
                    break;

                case "3":
                    SVMT.Join();
                    SVM.PredictSpam();
                    break;

                case "0":
                    string newInput = IO.ChangeInput();
                    File.WriteAllText(userInputPath, newInput);
                    Main(args);
                    break;

                default:
                    goto sorryforthislabel;
            }

            IO.PrintResult(prediction, userInput);
            Console.ReadLine();
            goto sorryforthislabel;
        }
    }
}