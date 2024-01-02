using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    internal class NaiveBayesClassifier
    {
        private Dictionary<string, double> spamProbabilities;
        private Dictionary<string, double> hamProbabilities;
        private double spamMessagesCount;
        private double hamMessagesCount;

        public NaiveBayesClassifier()
        {
            spamProbabilities = new Dictionary<string, double>();
            hamProbabilities = new Dictionary<string, double>();
            spamMessagesCount = 0;
            hamMessagesCount = 0;
        }

        public void Train(IEnumerable<IsSpamModel> data)
        {
            foreach (var item in data)
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

        public bool IsSpam(string message)
        {
            var words = message.Split(new char[] { ' ', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
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

        public static void PredictSpam(string userInputPath, string dataPath)
        {           
            NaiveBayesClassifier classifier = new NaiveBayesClassifier();
            
            var trainingData = LoadData(dataPath);
            
            classifier.Train(trainingData);
          
            string userInput = File.ReadAllText(userInputPath, Encoding.UTF8);
            
            bool isSpam = classifier.IsSpam(userInput);

            Console.WriteLine($"Сообщение: '{userInput}' {(isSpam ? "является спамом" : "не является спамом")}");



        }

        private static IEnumerable<IsSpamModel> LoadData(string dataPath)
        {
            var data = new List<IsSpamModel>();
            var lines = File.ReadAllLines(dataPath);

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

    }
}
