using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    class SupportVectorMachine : IPredictable
    {
        public static string ? DataPath { get; set; }

        public static string ? UserInput { get; set; }

        public static string ? ModelPath { get; set; }

        private double[][] _inputs; // Входные данные
        private int[] _labels; // Метки классов
        private double[] _alphas; // Параметры Лагранжа
        private double _bias = 0; // Смещение
        private const double C = 1.0; // Параметр регуляризации
        private const double tol = 1e-3; // Точность
        private const double eps = 1e-3; // Эпсилон
        private const int maxPasses = 5; // Максимальное количество проходов

        public SupportVectorMachine(string dataPath, string userInput, string modelPath)
        {
            DataPath = dataPath;
            UserInput = userInput;
            ModelPath = modelPath;
        }

        public void Train()
        {
            PreprocessTextData();

            int passes = 0;
            while (passes < maxPasses)
            {
                int numChangedAlphas = 0;
                for (int i = 0; i < _inputs.Length; i++)
                {
                    double Ei = DecisionFunction(_inputs[i]) - _labels[i];
                    if ((_labels[i] * Ei < -tol && _alphas[i] < C) || (_labels[i] * Ei > tol && _alphas[i] > 0))
                    {
                        int j = i;
                        while (j == i) j = new Random().Next(_inputs.Length);

                        double Ej = DecisionFunction(_inputs[j]) - _labels[j];
                        double oldAlphaI = _alphas[i];
                        double oldAlphaJ = _alphas[j];

                        double L, H;
                        if (_labels[i] != _labels[j])
                        {
                            L = Math.Max(0, _alphas[j] - _alphas[i]);
                            H = Math.Min(C, C + _alphas[j] - _alphas[i]);
                        }
                        else
                        {
                            L = Math.Max(0, _alphas[i] + _alphas[j] - C);
                            H = Math.Min(C, _alphas[i] + _alphas[j]);
                        }
                        if (L == H) continue;

                        double eta = 2 * Kernel(_inputs[i], _inputs[j]) - Kernel(_inputs[i], _inputs[i]) - Kernel(_inputs[j], _inputs[j]);
                        if (eta >= 0) continue;

                        _alphas[j] -= _labels[j] * (Ei - Ej) / eta;
                        _alphas[j] = Math.Min(H, Math.Max(L, _alphas[j]));
                        if (Math.Abs(_alphas[j] - oldAlphaJ) < eps) continue;

                        _alphas[i] += _labels[i] * _labels[j] * (oldAlphaJ - _alphas[j]);

                        double b1 = _bias - Ei - _labels[i] * (_alphas[i] - oldAlphaI) * Kernel(_inputs[i], _inputs[i])
                                         - _labels[j] * (_alphas[j] - oldAlphaJ) * Kernel(_inputs[i], _inputs[j]);
                        double b2 = _bias - Ej - _labels[i] * (_alphas[i] - oldAlphaI) * Kernel(_inputs[i], _inputs[j])
                                         - _labels[j] * (_alphas[j] - oldAlphaJ) * Kernel(_inputs[j], _inputs[j]);

                        if (0 < _alphas[i] && _alphas[i] < C) _bias = b1;
                        else if (0 < _alphas[j] && _alphas[j] < C) _bias = b2;
                        else _bias = (b1 + b2) / 2.0;

                        numChangedAlphas++;
                    }
                }
                if (numChangedAlphas == 0) passes++;
                else passes = 0;
            }
        }

        public void PreprocessTextData()
        {
            MLContext mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromTextFile<IsSpamModel>(DataPath, hasHeader: true, separatorChar: ';');

            var textFeaturizingEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(IsSpamModel.Text));
            ITransformer textTransformer = textFeaturizingEstimator.Fit(dataView);
            IDataView transformedData = textTransformer.Transform(dataView);

            var features = mlContext.Data.CreateEnumerable<TransformText>(transformedData, reuseRowObject: false)
                                         .ToArray();

            _inputs = features.Select(f => f.Features.Select(x => (double)x).ToArray())
                                        .ToArray();
            _labels = features.Select(f => f.IsSpam ? 1 : -1)
                              .ToArray();

            _alphas = new double[_inputs.Length];
        }

        public double DecisionFunction(double[] x)
        {
            double sum = _bias;
            for (int i = 0; i < _inputs.Length; i++)
            {
                sum += _alphas[i] * _labels[i] * Kernel(_inputs[i], x);
            }
            return sum;
        }

        private double Kernel(double[] x1, double[] x2)
        {
            if (x1.Length == x2.Length)
                return Enumerable.Range(0, x1.Length).Sum(i => x1[i] * x2[i]);

            else return 1;
        }

        public int Predict(double[] x)
        {
            return DecisionFunction(x) >= 0 ? 1 : -1;
        }

        public bool PredictSpam()
        {
            double[] newInput = PreprocessText();

            int prediction = Predict(newInput);

            if (prediction == -1)
                return false;
            else
                return true;
        }

        public double[] PreprocessText()
        {
            MLContext mlContext = new MLContext();

            var data = new[] { new IsSpamModel { Text = UserInput } };
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var textFeaturizingEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(IsSpamModel.Text));
            var textTransformer = textFeaturizingEstimator.Fit(dataView);
            var transformedData = textTransformer.Transform(dataView);

            var featureRow = mlContext.Data.CreateEnumerable<TransformText>(transformedData, reuseRowObject: false).FirstOrDefault();
            if (featureRow == null)
            {
                throw new InvalidOperationException("Не удалось преобразовать текст в вектор признаков.");
            }

            return featureRow.Features.Select(x => (double)x)
                                      .ToArray();
        }
    }
}