import random
import matplotlib.pyplot as plt

class Particula:
    def __init__(self, dimensiones, limites_pos, limites_vel):
        self.dimensiones = dimensiones
        self.limites_pos = limites_pos
        self.limites_vel = limites_vel

        self.posicion = [random.uniform(*limites_pos[i]) for i in range(dimensiones)]
        self.velocidad = [random.uniform(*limites_vel[i]) for i in range(dimensiones)]

        self.valor = self.evaluar(self.posicion)
        self.mejor_posicion = list(self.posicion)
        self.mejor_valor = self.valor

    def evaluar(self, posicion):
        return sum(x**2 for x in posicion)

    def __str__(self):
        return (f"Posición: {self.posicion}, "
                f"Velocidad: {self.velocidad}, "
                f"Valor: {self.valor:.4f}, "
                f"Mejor valor: {self.mejor_valor:.4f}")

# Parámetros
num_particulas = 20
dimensiones = 2
limites_pos = [(-10, 10)] * dimensiones
limites_vel = [(-2, 2)] * dimensiones

# Crear partículas
particulas = [Particula(dimensiones, limites_pos, limites_vel) for _ in range(num_particulas)]

# Mostrar en consola
for i, p in enumerate(particulas, 1):
    print(f"Partícula {i}: {p}")

# Graficar
x = [p.posicion[0] for p in particulas]
y = [p.posicion[1] for p in particulas]

plt.figure(figsize=(6, 6))
plt.scatter(x, y, c='blue', label='Posición actual', s=60, alpha=0.7, edgecolors='k')
plt.title("Partículas Generadas")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.xlim(limites_pos[0])
plt.ylim(limites_pos[1])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
