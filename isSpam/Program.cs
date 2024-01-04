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
        public static readonly string _dataPath = Path.Combine("..", "..", "..", "data", "data_one.csv");
        public static readonly string _userInputPath = Path.Combine("..", "..", "..", "User", "user.txt");
        public static readonly string _modelPath = Path.Combine("..", "..", "..", "data", "Model.zip");
        
        static void Main(string[] args)
        {
            Console.Clear();
            Console.WriteLine("Выберите алгоритм для проверки сообщения на спам:");
            Console.WriteLine("1. Логистическая регрессия");
            Console.WriteLine("2. Наивный байесовский классификатор");
            Console.WriteLine("3. Метод опорных векторов");
            string? choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    LogisticRegressionClassifier.TrainSpam(_dataPath, _modelPath);
                    LogisticRegressionClassifier.PredictSpam(_userInputPath, _modelPath);                  
                    break;
                case "2":
                    NaiveBayesClassifier.PredictSpam(_userInputPath, _dataPath);
                    break;
                case "3":
                    MLContext mlContext = new MLContext();
                    (double[][] inputs, int[] labels) = Svm.PreprocessTextData(_dataPath, mlContext);
                    Svm svm = new Svm(inputs, labels);
                    svm.SvmTraining(_userInputPath, _dataPath);
                    break;
                default:
                    Main(args);
                    break;
            }
        }
    }
}

