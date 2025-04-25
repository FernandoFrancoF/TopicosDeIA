import random
import math

# ==== Datos iniciales ====
ciudades = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (6, 4)]
num_particulas = 20
iteraciones = 30

# ==== Funciones base ====

def calcular_distancia(ruta):
    return sum(
        math.dist(ciudades[ruta[i]], ciudades[ruta[(i + 1) % len(ruta)]])
        for i in range(len(ruta))
    )

# ==== Métodos de partícula ====

def crear_particula():
    ruta = random.sample(range(len(ciudades)), len(ciudades))
    distancia = calcular_distancia(ruta)
    return {
        'ruta': ruta,
        'distancia': distancia,
        'mejor_ruta': ruta.copy(),
        'mejor_distancia': distancia
    }

def mover_particula(particula):
    i, j = random.sample(range(len(particula['ruta'])), 2)
    particula['ruta'][i], particula['ruta'][j] = particula['ruta'][j], particula['ruta'][i]
    particula['distancia'] = calcular_distancia(particula['ruta'])

def evaluar_particula(particula):
    if particula['distancia'] < particula['mejor_distancia']:
        particula['mejor_ruta'] = particula['ruta'].copy()
        particula['mejor_distancia'] = particula['distancia']

# ==== Métodos del enjambre ====

def crear_enjambre():
    return [crear_particula() for _ in range(num_particulas)]

def mover_enjambre(enjambre):
    for p in enjambre:
        mover_particula(p)

def evaluar_enjambre(enjambre):
    for p in enjambre:
        evaluar_particula(p)
    return min(enjambre, key=lambda p: p['mejor_distancia'])

# ==== Ejecución principal ====

enjambre = crear_enjambre()
mejor_global = min(enjambre, key=lambda p: p['mejor_distancia'])

for i in range(iteraciones):
    mover_enjambre(enjambre)
    mejor_actual = evaluar_enjambre(enjambre)

    if mejor_actual['mejor_distancia'] < mejor_global['mejor_distancia']:
        mejor_global = mejor_actual

    print(f"Iteración {i+1}: Mejor distancia = {mejor_global['mejor_distancia']:.2f}, Ruta: {mejor_global['mejor_ruta']}")

print("\n✅ Mejor ruta final encontrada:")
print("Ruta:", mejor_global['mejor_ruta'])
print(f"Distancia total: {mejor_global['mejor_distancia']:.2f}")


