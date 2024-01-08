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
        public static string Menu(string userInput)
        {
            Console.Clear();
            Console.WriteLine($"Текущее сообщение - '{userInput}'\n");
            Console.WriteLine("Выберите алгоритм для проверки сообщения на спам:");
            Console.WriteLine("1. Логистическая регрессия");
            Console.WriteLine("2. Наивный байесовский классификатор");
            Console.WriteLine("3. Метод опорных векторов");
            Console.WriteLine("\n\n0. Изменить сообщение");

            string? choice = Console.ReadLine();
            return choice;
        }

        public static void PrintResult(bool prediction, string userInput)
        {
            if (prediction)
                Console.WriteLine($"Сообщение '{userInput}' является спамом");
            else
                Console.WriteLine($"Сообщение '{userInput}' не является спамом");
        }

        public static string ChangeInput()
        {
            Console.WriteLine("Введите новое сообщение:\n");
            return Console.ReadLine() ?? "";
        }
    }
}