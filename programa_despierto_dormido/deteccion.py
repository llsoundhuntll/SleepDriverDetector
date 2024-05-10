# Importando Bibliotecas
import cv2
import numpy as np
import os
import vlc

from tqdm import tqdm
from keras.models import load_model


class SleepDriverDetector:

    def __init__(self):
        # Inicializando Variables
        os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')
        self.video_capture = cv2.VideoCapture(0)
        self.eyeLeft = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
        self.eyeRight = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.model = load_model('keras_model.h5')
        self.p = vlc.MediaPlayer("wakeup.mp3")
        # Iniciando programa
        self.detect_sleep()

    def update_progress_bar(self, count, limite):
        bar_length = 30
        progress = int((count / limite) * bar_length)
        bar = "[" + "=" * progress + " " * (bar_length - progress) + "]"
        tqdm.write("Estado Dormido: {} ({:.2f}%)".format(bar, (count / limite) * 100))

    def play_sound(self):
        self.p.play()

    def stop_sound(self):
        self.p.stop()

    def detect_sleep(self):
        left_x, left_y, left_w, left_h = 0, 0, 0, 0
        right_x, right_y, right_w, right_h = 0, 0, 0, 0
        contador = 0
        limite = 15
        while True:
            ret, frame = self.video_capture.read()
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Mostrando Contador
            cv2.rectangle(frame, (0, 0), (width, int(height * 0.1)), (0, 0, 0), -1)
            # Texto en la imagen cv2.putText(frame, 'Contador: ' + str(contador), (int(width * 0.65), int(height *
            # 0.08)), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2)

            # Identificando el Ojo Derecho
            ojo_der = self.eyeRight.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=3,
                minSize=(30, 30)
            )
            for (x, y, w, h) in ojo_der:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                right_x, right_y, right_w, right_h = x, y, w, h
                break

            # Identificando el Ojo Izquierdo
            ojo_izq = self.eyeLeft.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=3,
                minSize=(30, 30)
            )
            for (x, y, w, h) in ojo_izq:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                left_x, left_y, left_w, left_h = x, y, w, h
                break

            # Identificando Coordenadas x,y iniciales y finales para extraer la foto de los ojos
            if left_x > right_x:
                start_x, end_x = right_x, (left_x + left_w)
            else:
                start_x, end_x = left_x, (right_x + right_w)

            if left_y > right_y:
                start_y, end_y = right_y, (left_y + left_h)
            else:
                start_y, end_y = left_y, (right_y + right_h)

            # Algoritmo de deteccion de sueño
            if ((end_x - start_x) > 120 and (end_y - start_y) < 200):
                start_x, start_y, end_x, end_y = start_x - 30, start_y - 50, end_x + 30, end_y + 50
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                img = frame[start_y:end_y, start_x:end_x]
                imagen = cv2.resize(img, (224, 224))
                imagen_normalizada = (imagen.astype(np.float32) / 127.0) - 1
                self.data[0] = imagen_normalizada
                prediction = self.model.predict(self.data, verbose=0)

                if list(prediction)[0][1] >= 0.95:
                    # Texto en la imagen
                    # cv2.putText(frame,'Durmiendo : '+str(round(list(prediction)[0][1],3)),(10, int(height*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                    contador += 1
                    self.update_progress_bar(contador, limite)

                if list(prediction)[0][0] >= 0.95:
                    # Texto en la imagen
                    # cv2.putText(frame,'Despierto',(10, int(height*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
                    contador -= 1
                    self.update_progress_bar(contador, limite)

                if contador >= limite:
                    contador = limite
                    self.play_sound()

                elif 0 < contador < limite:
                    self.stop_sound()
                elif contador < 0:
                    contador = 0

            # Mostrar el video de la webcam
            cv2.imshow('video', frame)

            # Tecla de salida - acaba la transmisión
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.video_capture.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                contador = 0
                self.p.stop()

    @classmethod
    def _build(cls):
        cls()


SleepDriverDetector._build()
