using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    public class LogisticRegressionClassifier : IPredictable
    {
        public static string ? DataPath { get; set; }

        public static string ? UserInput { get; set; }

        public static string ? ModelPath { get; set; }

        public LogisticRegressionClassifier(string dataPath, string userInput, string modelPath)
        {
            DataPath = dataPath;
            UserInput = userInput;
            ModelPath = modelPath;
        }


        public bool PredictSpam()
        {
            MLContext mlContext = new MLContext();

            ITransformer model = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            var predictor = mlContext.Model.CreatePredictionEngine<IsSpamModel, PredictionsModel>(model);

            var input = new IsSpamModel { Text = UserInput };

            var prediction = predictor.Predict(input);

            return prediction.Prediction;
        }

        public void Train()
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<IsSpamModel>(DataPath, hasHeader: true, separatorChar: ';');

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(IsSpamModel.Text))
                                                    .Append(mlContext.BinaryClassification.Trainers
                                                    .LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            ITransformer model = pipeline.Fit(dataView);

            mlContext.Model.Save(model, dataView.Schema, ModelPath);
        }
    }
}