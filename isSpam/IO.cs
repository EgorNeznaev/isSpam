using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace isSpam
{
    internal class IO
    {
        public static string Menu()
        {
            Console.Clear();
            Console.WriteLine("Выберите алгоритм для проверки сообщения на спам:");
            Console.WriteLine("1. Логистическая регрессия");
            Console.WriteLine("2. Наивный байесовский классификатор");
            Console.WriteLine("3. Метод опорных векторов");

            string? choice = Console.ReadLine();
            return choice;
        }

        public static void PrintResult(bool prediction)
        {
            if (prediction)
                Console.WriteLine($"Сообщение является спамом");
            else
                Console.WriteLine($"Сообщение не является спамом"); ;
        }
    }
}
