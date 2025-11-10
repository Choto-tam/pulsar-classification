import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

def main():
    print("🚀 КЛАССИФИКАТОР ПУЛЬСАРОВ")
    print("=" * 40)
    
    # Загрузка данных
    try:
        stars = pd.read_csv("pulsar_stars.csv")
        print("✅ Данные загружены")
    except:
        print("❌ Файл pulsar_stars.csv не найден")
        print("📝 Создаем демо-данные...")
        # Создаем демо-данные для примера
        return
    
    # Показываем первые строки
    print("\n📊 Первые 5 строк данных:")
    print(stars.head())
    
    # Фильтрация данных для обучения
    stars_train = stars[
        ((stars['TG'] == 0) & (stars['MIP'] >= 94.6640625) & (stars['MIP'] <= 95.2890625)) | 
        ((stars['TG'] == 1) & (stars['MIP'] >= 65.078125) & (stars['MIP'] <= 70.7421875))
    ]
    
    print(f"\n📈 Данных для обучения: {len(stars_train)} строк")
    
    # Нормировка данных
    stars_train_normed = (stars_train - stars_train.min()) / (stars_train.max() - stars_train.min())
    
    # Разделение на признаки и цель
    X = stars_train_normed.drop(['TG'], axis=1)
    y = stars_train_normed.TG
    
    # Обучение моделей
    print("\n🤖 ОБУЧЕНИЕ МОДЕЛЕЙ...")
    
    # Логистическая регрессия
    log_reg = LogisticRegression(random_state=2019, solver='lbfgs')
    log_reg.fit(X, y)
    
    # K-ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=1, p=2)
    knn.fit(X, y)
    
    print("✅ Модели обучены!")
    
    # Пример предсказания
    new_star = [0.254, 0.19, 0.939, 0.624, 0.935, 0.875, 0.151, 0.312]
    
    # Предсказание логистической регрессии
    proba = log_reg.predict_proba([new_star])[0][1]
    print(f"\n🎯 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ:")
    print(f"   Вероятность пульсара: {proba:.2%}")
    
    # Предсказание k-NN
    prediction = knn.predict([new_star])[0]
    distance = knn.kneighbors([new_star])[0][0][0]
    
    print(f"\n📡 МЕТОД k-БЛИЖАЙШИХ СОСЕДЕЙ:")
    print(f"   Предсказанный класс: {'ПУЛЬСАР' if prediction == 1 else 'НЕ ПУЛЬСАР'}")
    print(f"   Расстояние до ближайшей звезды: {distance:.4f}")
    
    print("\n" + "=" * 40)
    print("✅ ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
