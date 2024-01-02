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
        static readonly string _dataPath = @"C:\Users\Nezna\OneDrive\Desktop\isSpam\isSpam\data\data_one.csv";
        static readonly string _userInputPath = @"C:\Users\Nezna\OneDrive\Desktop\isSpam\isSpam\User\user.txt";
        static readonly string _modelPath = @"C:\Users\Nezna\OneDrive\Desktop\isSpam\isSpam\data\Model.zip";
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            var textLoader = mlContext.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("IsSpam", DataKind.Boolean, 0),
                new TextLoader.Column("Text", DataKind.String, 1)
            }, hasHeader: true, separatorChar: ';');

            IDataView dataView = textLoader.Load(_dataPath);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(IsSpamModel.Text))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "IsSpam", featureColumnName: "Features"));

            // Обучение модели
            var model = pipeline.Fit(dataView);

            // Сохранение модели
            mlContext.Model.Save(model, dataView.Schema, _modelPath);

            // Предсказание для нового сообщения
            PredictSpam(mlContext, model);
        }
        public static void PredictSpam(MLContext mlContext, ITransformer model)
        {
            var predictor = mlContext.Model.CreatePredictionEngine<IsSpamModel, PredictionsModel>(model);

            // Read user input with UTF-8 encoding
            string userInput = File.ReadAllText(_userInputPath, Encoding.UTF8);
            var input = new IsSpamModel { Text = userInput };

            // Prediction
            var prediction = predictor.Predict(input);

            // Output the result
            Console.WriteLine($"Сообщение: '{input.Text}' {(prediction.Prediction ? "является спамом" : "не является спамом")}");
        }
    }
}

