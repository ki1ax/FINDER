from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import dlib
import numpy as np
from skimage import io
import nmslib
import os

app = Flask(__name__)

# Загрузка моделей
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()


def get_face_descriptor(img, detector, sp, face_rec):
    # Обнаружение лиц и получение дескриптора
    detected_faces = detector(img, 1)
    if not detected_faces:
        raise ValueError("На фотографии не распознаны лица.")
    if len(detected_faces) > 1:
        raise ValueError("На фотографии должно быть одно лицо.")
    for k, d in enumerate(detected_faces):
        shape = sp(img, d)
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        face_descriptor = np.asarray(face_descriptor)
        return face_descriptor
    raise ValueError("Дескриптор лица не может быть вычислен.")


def print_id(n, ids, associations):
    # Печать ID пользователя VK
    best_dx = ids[n]
    s = associations.get(best_dx, '')
    if s:
        s = 'https://vk.com/' + s.split('_')[0]
        for bad_symbols in ['.txt', '.npy', '\n']:
            s = s.replace(bad_symbols, '')
        return s
    else:
        return "Связи для идентификатора не найдены: " + str(best_dx)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        uploaded_file = request.files['photo']
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join('uploads', filename)
            uploaded_file.save(file_path)
            img = io.imread(file_path)
            try:
                embedding = get_face_descriptor(img, detector, sp, face_rec)
            except ValueError as e:  # Обработка исключения, если лицо не найдено
                os.remove(file_path)  # Удаление файла после обработки
                return jsonify({'error': str(e)}), 400  # Возврат ошибки на страницу

            # Инициализация nmslib и загрузка индекса
            index = nmslib.init(method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR)
            index.loadIndex('embeddings.bin')
            query_time_params = {'efSearch': 400}
            index.setQueryTimeParams(query_time_params)

            # Чтение файла ассоциаций один раз
            associations = {}
            with open('associations.txt', 'r') as file_:
                for line in file_:
                    key, value = line.strip().split('|')
                    associations[int(key)] = value

            # Обработка изображения и поиск ближайших соседей
            ids, dists = index.knnQuery(embedding, k=5)
            results = []
            for i in range(5):
                result = print_id(i, ids, associations)
                results.append(result)

            return jsonify(results)
        else:
            # Возвращаем сообщение об ошибке, если формат файла не поддерживается
            error_message = 'Неправильный формат файла. Пожалуйста, загрузите фото в формате JPEG или JPG.'
            return render_template('index.html', error=error_message)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg'}


if __name__ == '__main__':
    app.run(debug=False)  # Отключение режима отладки для развертывания