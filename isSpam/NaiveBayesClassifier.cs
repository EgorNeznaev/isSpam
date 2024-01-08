using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace isSpam
{
    class NaiveBayesClassifier : IPredictable
    {
        public static string ? DataPath { get; set; }

        public static string ? UserInput { get; set; }

        public static string ? ModelPath { get; set; }

        private Dictionary<string, double> spamProbabilities;
        private double spamMessagesCount;

        private Dictionary<string, double> hamProbabilities;
        private double hamMessagesCount;

        public NaiveBayesClassifier(string dataPath, string userInput, string modelPath)
        {
            DataPath = dataPath;
            UserInput = userInput;
            ModelPath = modelPath;

            spamProbabilities = new Dictionary<string, double>();
            hamProbabilities = new Dictionary<string, double>();

            spamMessagesCount = 0;
            hamMessagesCount = 0;
        }

        public void Train()
        {
            var trainingData = LoadData();

            foreach (var item in trainingData)
            {
                var words = item.Text.Split(new char[] { ' ', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var word in words)
                {
                    if (item.IsSpam)
                    {
                        spamProbabilities[word] = spamProbabilities.ContainsKey(word) ? spamProbabilities[word] + 1 : 1;
                        spamMessagesCount++;
                    }
                    else
                    {
                        hamProbabilities[word] = hamProbabilities.ContainsKey(word) ? hamProbabilities[word] + 1 : 1;
                        hamMessagesCount++;
                    }
                }
            }

            foreach (var word in spamProbabilities.Keys.ToList())
            {
                spamProbabilities[word] = (spamProbabilities[word] + 1) / (spamMessagesCount + 2);
            }

            foreach (var word in hamProbabilities.Keys.ToList())
            {
                hamProbabilities[word] = (hamProbabilities[word] + 1) / (hamMessagesCount + 2);
            }
        }

        private static IEnumerable<IsSpamModel> LoadData()
        {
            var data = new List<IsSpamModel>();
            var lines = File.ReadAllLines(DataPath);

            foreach (var line in lines.Skip(1))
            {
                var columns = line.Split(';');
                if (columns.Length == 2)
                {
                    data.Add(new IsSpamModel
                    {
                        IsSpam = columns[0] == "1",
                        Text = columns[1]
                    });
                }
            }

            return data;
        }

        public bool PredictSpam()
        {
            var words = UserInput.Split(new char[] { ' ', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

            double spamLogProbability = Math.Log(spamMessagesCount / (spamMessagesCount + hamMessagesCount));
            double hamLogProbability = Math.Log(hamMessagesCount / (spamMessagesCount + hamMessagesCount));

            foreach (var word in words)
            {
                if (spamProbabilities.ContainsKey(word))
                    spamLogProbability += Math.Log(spamProbabilities[word]);
                else
                    spamLogProbability += Math.Log(1 / (spamMessagesCount + 2));

                if (hamProbabilities.ContainsKey(word))
                    hamLogProbability += Math.Log(hamProbabilities[word]);
                else
                    hamLogProbability += Math.Log(1 / (hamMessagesCount + 2));
            }

            return spamLogProbability > hamLogProbability;
        }               
    }
}