import pandas as pd
import argparse

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Конвертация CSV в Excel')
    parser.add_argument('input', help='Путь к входному CSV файлу')
    parser.add_argument('output', help='Путь к выходному Excel файлу')
    args = parser.parse_args()

    # Читаем CSV файл
    print(f"Читаем файл: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Загружено записей: {len(df)}")

    # Сохраняем в Excel
    print(f"Сохраняем в: {args.output}")
    df.to_excel(args.output, index=False)
    print("Конвертация завершена")

    # Выводим список колонок для проверки
    print("\nСписок колонок в файле:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main() 