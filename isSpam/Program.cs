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
/*        public static readonly string _dataPath = @"C:\Users\Nezna\OneDrive\Desktop\isSpam\isSpam\data\data_one.csv";
        public static readonly string _userInputPath = @"C:\Users\Nezna\OneDrive\Desktop\isSpam\isSpam\User\user.txt";
        public static readonly string _modelPath = @"C:\Users\Nezna\OneDrive\Desktop\isSpam\isSpam\data\Model.zip";*/
        public static readonly string _dataPath = @"C:\Users\1PCHome\source\repos\isSpam\isSpam\data\data_one.csv";
        public static readonly string _userInputPath = @"C:\Users\1PCHome\source\repos\isSpam\isSpam\User\user.txt";
        public static readonly string _modelPath = @"C:\Users\1PCHome\source\repos\isSpam\isSpam\data\Model.zip";
        static void Main(string[] args)
        {
            Console.WriteLine("Выберите алгоритм для проверки сообщения на спам:");
            Console.WriteLine("1.Логистическая регрессия");
            Console.WriteLine("2.Наивный байесовский классификатор");
            Console.WriteLine("3.Классификация данных методом опорных векторов");
            string choice = Console.ReadLine();

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
                    Svm.SvmTraining();
                default:
                    Console.WriteLine("Неверный выбор. Пожалуйста, введите 1 или 2.");
                    break;
            }
        }
    }
}

