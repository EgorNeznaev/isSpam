using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    public class Svm
    {
        private double[][] inputs; // Входные данные
        private int[] labels; // Метки классов
        private double[] alphas; // Параметры Лагранжа
        private double b; // Смещение
        private double C = 1.0; // Параметр регуляризации
        private double tol = 1e-3; // Точность
        private double eps = 1e-3; // Эпсилон
        private int maxPasses = 5; // Максимальное количество проходов

        public Svm(double[][] inputs, int[] labels)
        {
            this.inputs = inputs;
            this.labels = labels;
            alphas = new double[inputs.Length];
            b = 0;
        }

        private double Kernel(double[] x1, double[] x2)
        {
            // Линейное ядро
            return Enumerable.Range(0, x1.Length).Sum(i => x1[i] * x2[i]);
        }

        public void Train()
        {
            int passes = 0;
            while (passes < maxPasses)
            {
                int numChangedAlphas = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    double Ei = DecisionFunction(inputs[i]) - labels[i];
                    if ((labels[i] * Ei < -tol && alphas[i] < C) || (labels[i] * Ei > tol && alphas[i] > 0))
                    {
                        // Выбираем случайный j, не равный i
                        int j = i;
                        while (j == i) j = new Random().Next(inputs.Length);

                        double Ej = DecisionFunction(inputs[j]) - labels[j];
                        double oldAlphaI = alphas[i];
                        double oldAlphaJ = alphas[j];

                        // Вычисляем границы для alphas[j]
                        double L, H;
                        if (labels[i] != labels[j])
                        {
                            L = Math.Max(0, alphas[j] - alphas[i]);
                            H = Math.Min(C, C + alphas[j] - alphas[i]);
                        }
                        else
                        {
                            L = Math.Max(0, alphas[i] + alphas[j] - C);
                            H = Math.Min(C, alphas[i] + alphas[j]);
                        }
                        if (L == H) continue;

                        // Вычисляем эта
                        double eta = 2 * Kernel(inputs[i], inputs[j]) - Kernel(inputs[i], inputs[i]) - Kernel(inputs[j], inputs[j]);
                        if (eta >= 0) continue;

                        // Обновляем alphas[j]
                        alphas[j] -= labels[j] * (Ei - Ej) / eta;
                        alphas[j] = Math.Min(H, Math.Max(L, alphas[j]));
                        if (Math.Abs(alphas[j] - oldAlphaJ) < eps) continue;

                        // Обновляем alphas[i]
                        alphas[i] += labels[i] * labels[j] * (oldAlphaJ - alphas[j]);

                        // Вычисляем b1 и b2
                        double b1 = b - Ei - labels[i] * (alphas[i] - oldAlphaI) * Kernel(inputs[i], inputs[i])
                                         - labels[j] * (alphas[j] - oldAlphaJ) * Kernel(inputs[i], inputs[j]);
                        double b2 = b - Ej - labels[i] * (alphas[i] - oldAlphaI) * Kernel(inputs[i], inputs[j])
                                         - labels[j] * (alphas[j] - oldAlphaJ) * Kernel(inputs[j], inputs[j]);

                        // Вычисляем b
                        if (0 < alphas[i] && alphas[i] < C) b = b1;
                        else if (0 < alphas[j] && alphas[j] < C) b = b2;
                        else b = (b1 + b2) / 2.0;

                        numChangedAlphas++;
                    }
                }
                if (numChangedAlphas == 0) passes++;
                else passes = 0;
            }
        }

        public double DecisionFunction(double[] x)
        {
            double sum = b;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += alphas[i] * labels[i] * Kernel(inputs[i], x);
            }
            return sum;
        }

        public int Predict(double[] x)
        {
            return DecisionFunction(x) >= 0 ? 1 : -1;
        }
        // Использование
        public void SvmTraining(string userInputPath, string dataPath)
        {
            (double[][] inputs, int[] labels) = PreprocessTextData(dataPath, new MLContext());
            

            Svm svm = new Svm(inputs, labels);
            svm.Train();

            PredictFromFile(userInputPath, svm);
        }

        public void PredictFromFile(string userInputPath, Svm svm)
        {
            string userText = File.ReadAllText(userInputPath);
            double[] newInput = PreprocessText(userText, new MLContext());

            int prediction = svm.Predict(newInput);
            Console.WriteLine($"Предсказание: {prediction}");
        }

        public double[] PreprocessText(string text, MLContext mlContext)
        {
            // Создаем одну строку данных
            var data = new[] { new IsSpamModel { Text = text } };
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Преобразование текста в числовые векторы с помощью TF-IDF
            var textFeaturizingEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(IsSpamModel.Text));
            ITransformer textTransformer = textFeaturizingEstimator.Fit(dataView);
            IDataView transformedData = textTransformer.Transform(dataView);

            // Извлечение вектора признаков
            var featureRow = mlContext.Data.CreateEnumerable<TransformText>(transformedData, reuseRowObject: false).FirstOrDefault();
            if (featureRow == null)
            {
                throw new InvalidOperationException("Не удалось преобразовать текст в вектор признаков.");
            }

            return featureRow.Features.Select(x => (double)x).ToArray();
        }

        public static (double[][], int[]) PreprocessTextData(string csvFilePath, MLContext mlContext)
        {
            // Загрузка данных из CSV-файла
            IDataView dataView = mlContext.Data.LoadFromTextFile<IsSpamModel>(csvFilePath, hasHeader: true, separatorChar: ';');

            // Преобразование столбца 'Text' в числовые векторы с помощью TF-IDF
            var textFeaturizingEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(IsSpamModel.Text));
            ITransformer textTransformer = textFeaturizingEstimator.Fit(dataView);
            IDataView transformedData = textTransformer.Transform(dataView);

            // Извлечение векторов и меток классов
            var features = mlContext.Data.CreateEnumerable<TransformText>(transformedData, reuseRowObject: false).ToArray();
            double[][] inputs = features.Select(f => f.Features.Select(x => (double)x).ToArray()).ToArray();
            int[] labels = features.Select(f => f.IsSpam ? 1 : -1).ToArray();

            return (inputs, labels);
        }
    }
}
