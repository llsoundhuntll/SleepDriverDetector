from tqdm import tqdm
import time

# Ejemplo de contador
contador = 0
limite = 15

# Ciclo para simular el incremento del contador
for i in tqdm(range(limite), desc='Progreso del contador', unit='%', ncols=100):
    # Simular un incremento del contador
    contador += 1
    # Simular un tiempo de espera entre incrementos
    time.sleep(0.5)  # Esto puede ser cualquier operación que realices en tu script

print("Contador alcanzó el límite.")