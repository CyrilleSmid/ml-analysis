import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import time

# Создаем новую модель НС и обучаем
def create_and_train_model(in_data_train, out_data_train):
    # При запуске возможна ошибка, когда Keras требует более новую версию TensorFlow:
    # Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`
    # в данном случае использовать pip install --user --upgrade tensorflow и перезапустить среду разработки
    model = Sequential() # Модель НС - сеть прямого распространения
    model.add(Input(shape=(in_data_train.shape[1],))) # входной слой
    # Добавляем 2 полносвязных (Dense) слоя с сигмоидальной функцией активации
    model.add(Dense(5, activation="sigmoid")) # промежуточный слой (5 нейронов)
    model.add(Dense(1, activation="sigmoid")) # выходной слой (1 нейрон)

    # Компилируем модель и устанавливаем параметры оптимизатора весов (алгоритма обучения)
    model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.RMSprop(learning_rate=0.005))

    start_time = time.time() # текущее время (старт обучения)
    # Обучение: epochs - количество эпох (циклов), batch_size - размер набора обучающих данных для корреции весов
    history = model.fit(in_data_train, out_data_train, epochs=1000, batch_size=32)
    # Вывод затраченного на обучение времени
    print("--- %s seconds ---" % (time.time() - start_time))
    # Вывод структуры модели
    print("Input size: ", in_data_train.shape[1])
    model.summary()

    return model

# Сохраняем структуру модели в файл model.json, а весовые коэффициенты в weights.h5
def save_model(model):
    json_file = 'model.json'
    model_json = model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)

    model.save_weights('weights.h5')

# Загружаем модель из файла json_file, а веса из weights.h5
def load_model(json_file, weights_file):
    with open(json_file, 'r') as f:
        loaded_model = model_from_json(f.read())

    loaded_model.load_weights(weights_file)

    return loaded_model

if __name__ == '__main__':
    dataset = pd.read_csv("apartments.csv", sep=';', header='infer', names=None, encoding="utf-8")  # возвращает Pandas DataFrame
    pd.set_option("display.max_rows", None, "display.max_columns", None)  # Настройки вывода таблицы
    print(dataset.shape)  # Выводим размерность данных
    # print(dataset.head(10)) # Выводим первые 10 строк DataFrame (проверяем, что файл загрузился верно)

    in_data = dataset.iloc[:,1:10]  # .values # выбираем все строки c 2 по 10 колонок и трансформируем в np-массив
    out_data = dataset.iloc[:, 10:11].values  # выбираем все строки (последняя колонка) и трансформируем в np-массив

    print(in_data.shape)
    print(out_data.shape)
    # Нормализация данных - линейная нормализация Xnorm = (Xi - Xmin)/(Xmax - Xmin)) -> [0...1]
    norm = MinMaxScaler()       # Нормализатор для входных данных
    norm_out = MinMaxScaler()   # Нормализатор для выходных данных
    out_data = norm_out.fit_transform(out_data)     # Нормализуем выходные значения

    # One-hot (бинарное) кодирование - приведение данных к битовому формату, например:
    # "красный" - 1 0 0
    # "зеленый" - 0 1 0
    # "синий" - 0 0 1
    # Названия колонок для one-hot кодирования
    one_hot_cols = ['Район', 'Тип планировки']
    # Используем встроенный в Pandas метод one-hot кодирования get_dummies
    for col_name in one_hot_cols:
        one_hot = pd.get_dummies(in_data[col_name])
        in_data = in_data.drop(col_name, axis=1)
        in_data = in_data.join(one_hot)

    # Трансформируем колонки с True и False значениями в 1 и 0 соответственно
    bin_cols = ['Первый/Последний этаж', 'Наличие агенства']
    for col_name in bin_cols:
        in_data[col_name] = in_data[col_name].astype(int)

    # Выводим первые 10 строк нормализованных данных для проверки
    print(in_data.head(10))

    # Делаем MinMax-нормализацию входных значений
    in_data = norm.fit_transform(in_data.values)
    print("Размерность входных данных: ", in_data.shape)

    # Разделяем из общего набора данных тренировочную (90%) и тестовую выборку (10%)
    in_data_train, in_data_test, out_data_train, out_data_test = train_test_split(in_data, out_data, test_size=0.1)

    # Создаем новую модель НС и обучаем на тренировочной выборке
    model = create_and_train_model(in_data_train, out_data_train)
    # Если необходимо загрузить сохраненную модель из файла, раскомментируйте строку ниже и закомментируйте строку выше
    #model = load_model('model.json', 'weights.h5')
    # Прогоняем через обученную модель тестовый набор данных для оценки точности
    out_pred = model.predict(in_data_test)

    # Полученные (out_pred) и соответствующие им эталонные значения (out_data_test) помещаем в стандартные списки
    predicted = list()
    for i in range(len(out_pred)):
        predicted.append(out_pred[i][0])

    test = list()
    for i in range(len(out_data_test)):
        test.append(out_data_test[i][0])
    # Вычисляем относительную ошибку
    approx_err = mean_absolute_percentage_error(predicted, test)
    print('Approximation error:', approx_err*100)

    # Раскомментируйте строку ниже, если хотите вывести выходные значения сети,
    # приведенные к РЕАЛЬНЫМ величинам (выполняется операция обратная нормализации)
    # print(norm_out.inverse_transform(out_data_test))

    print("Save model to file ?:")
    q = input()
    if q.lower() == 'y':
        save_model(model)



