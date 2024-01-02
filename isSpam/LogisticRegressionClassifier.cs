using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    internal class LogisticRegressionClassifier: Program
    {
        public static void TrainSpam(string dataPath, string modelPath)
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<IsSpamModel>(dataPath, hasHeader: true, separatorChar: ';');

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(IsSpamModel.Text))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            ITransformer model = pipeline.Fit(dataView);

            mlContext.Model.Save(model, dataView.Schema, modelPath);
           
        }

        public static void PredictSpam(string userInputPath, string modelPath)
        {
            MLContext mlContext = new MLContext();

            ITransformer model = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<IsSpamModel, PredictionsModel>(model);

            string userInput = File.ReadAllText(userInputPath, Encoding.UTF8);
            var input = new IsSpamModel { Text = userInput };

            var prediction = predictor.Predict(input);

            Console.WriteLine($"Сообщение: '{input.Text}' {(prediction.Prediction ? "является спамом" : "не является спамом")}");
        }
    }
}
