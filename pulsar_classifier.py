import pandas as pd
import warnings
warnings.filterwarnings("ignore")
stars = pd.read_csv("pulsar_stars_new.csv")

print(stars.head())

stars_train = stars[((stars['TG'] == 0) & (stars['MIP'] >= 94.6640625) & (stars['MIP'] <= 95.2890625)) | ((stars['TG'] == 1) & (stars['MIP'] >= 65.078125) & (stars['MIP'] <= 70.7421875))]
print(stars_train)

# Среднее столбца MIP
stars_train.MIP.mean()
print(stars_train.MIP.mean())

# Выполним линейную нормировку
stars_train_normed = (stars_train - stars_train.min())/(stars_train.max() - stars_train.min())
print(stars_train_normed)

# Среднее столбца MIP после нормировки
stars_train_normed.MIP.mean()
print(stars_train_normed.MIP.mean())

# Предикторы
X = pd.DataFrame(stars_train_normed.drop(['TG'], axis=1))
# Отклики
y = stars_train_normed.TG

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, y)

new_star = [0.254, 0.19, 0.939, 0.624, 0.935, 0.875, 0.151, 0.312]
clf.predict_proba([new_star])
print('Вероятность отнесения к классу пульсар: ', clf.predict_proba([new_star])[0][1])

clf.predict([new_star])

from sklearn.neighbors import KNeighborsClassifier
# Создадим объект класса KNeighborClassifier
neigh = KNeighborsClassifier(n_neighbors = 1, p = 2)
# Обучаем классификатор на тренировочных данных
neigh.fit(X, y)

print('Предсказанный класс: ', neigh.predict([new_star])[0])
print('Расстояние до ближайшей звезды: ', neigh.kneighbors([new_star])[0][0][0])
  
