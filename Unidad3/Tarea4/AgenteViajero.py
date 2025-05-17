import random
import matplotlib.pyplot as plt
import networkx as nx
from deap import base, creator, tools, algorithms

# Distancias entre ciudades (origen, destino): distancia en km
distancias_ciudades = {
    ("Bilbao", "Celta"): 378, ("Bilbao", "Zaragoza"): 324,
    ("Celta", "Vigo"): 171, ("Celta", "Valladolid"): 235,
    ("Vigo", "Valladolid"): 356, ("Vigo", "Sevilla"): 245,
    ("Valladolid", "Madrid"): 193, ("Valladolid", "Zaragoza"): 390,
    ("Valladolid", "Jaen"): 411,
    ("Zaragoza", "Madrid"): 190, ("Zaragoza", "Albacete"): 215,
    ("Zaragoza", "Barcelona"): 296, ("Zaragoza", "Gerona"): 289,
    ("Madrid", "Albacete"): 251,
    ("Albacete", "Valencia"): 191, ("Albacete", "Granada"): 244,
    ("Albacete", "Murcia"): 150,
    ("Murcia", "Granada"): 257, ("Murcia", "Valencia"): 241,
    ("Barcelona", "Gerona"): 100, ("Barcelona", "Valencia"): 349,
    ("Granada", "Jaen"): 207, ("Granada", "Sevilla"): 211,
    ("Jaen", "Sevilla"): 125,
}

# Extraemos las ciudades únicas
ciudades_unicas = list({ciudad for ruta in distancias_ciudades.keys() for ciudad in ruta})

# Mapas para conversión rápida entre índices y nombres
id_a_ciudad = {idx: ciudad for idx, ciudad in enumerate(ciudades_unicas)}
ciudad_a_id = {ciudad: idx for idx, ciudad in id_a_ciudad.items()}

# Creamos un diccionario con distancias usando índices en lugar de nombres
distancias_indices = {}
for (origen, destino), distancia in distancias_ciudades.items():
    distancias_indices[(ciudad_a_id[origen], ciudad_a_id[destino])] = distancia

def calcular_distancia_total(ruta):
    """Calcula la suma total de distancias para una ruta dada (lista de índices)."""
    suma = 0
    n = len(ruta)
    for i in range(n):
        ciudad_actual = ruta[i]
        ciudad_siguiente = ruta[(i + 1) % n]
        if (ciudad_actual, ciudad_siguiente) in distancias_indices:
            suma += distancias_indices[(ciudad_actual, ciudad_siguiente)]
        elif (ciudad_siguiente, ciudad_actual) in distancias_indices:
            suma += distancias_indices[(ciudad_siguiente, ciudad_actual)]
        else:
            suma += float('inf')  # Camino no válido
    return suma

# Configuración DEAP para optimización con algoritmo genético
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("perm_indices", random.sample, range(len(ciudades_unicas)), len(ciudades_unicas))
toolbox.register("individual", tools.initIterate, creator.Individuo, toolbox.perm_indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registro con nombres que DEAP espera para algoritmos como eaSimple
toolbox.register("evaluate", lambda ind: (calcular_distancia_total(ind),))
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def dibujar_ruta_optima(ruta_indices):
    """Visualiza la ruta óptima sobre un grafo de ciudades."""
    ruta_nombres = [id_a_ciudad[i] for i in ruta_indices]

    G = nx.Graph()
    for (origen, destino), dist in distancias_ciudades.items():
        G.add_edge(origen, destino, weight=dist)

    # Usar layout circular para darle otro look
    posiciones = nx.circular_layout(G)

    plt.figure(figsize=(14, 10))
    
    # Nodos en color verde oscuro con borde negro y tamaño mayor
    nx.draw_networkx_nodes(G, posiciones, node_color='#2E8B57', edgecolors='black', node_size=1600)
    
    # Etiquetas con fondo blanco para mejor visibilidad
    nx.draw_networkx_labels(G, posiciones, font_size=12, font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    # Dibujar todas las aristas en gris claro y más delgadas
    nx.draw_networkx_edges(G, posiciones, edge_color='lightgray', width=1.5)
    
    # Etiquetas con distancia en negrita y color azul
    etiquetas_aristas = {(u, v): f"{d['weight']} km" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, posiciones, edge_labels=etiquetas_aristas, font_color='blue', font_weight='bold', font_size=10)

    # Crear lista de aristas para la ruta óptima
    aristas_ruta = [(ruta_nombres[i], ruta_nombres[(i + 1) % len(ruta_nombres)]) for i in range(len(ruta_nombres))]

    # Validar direcciones para grafo no dirigido
    aristas_ruta_validas = []
    for u, v in aristas_ruta:
        if G.has_edge(u, v):
            aristas_ruta_validas.append((u, v))
        elif G.has_edge(v, u):
            aristas_ruta_validas.append((v, u))

    # Dibujar la ruta con línea punteada roja y ancho mayor, con flechas para indicar dirección
    nx.draw_networkx_edges(
        G, posiciones,
        edgelist=aristas_ruta_validas,
        edge_color='red',
        style='dashed',
        width=4,
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>'
    )

    plt.title("Ruta Óptima - Agente Viajero", fontsize=18, fontweight='bold', color='#4B0082')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def ejecutar_algoritmo():
    random.seed(42)
    poblacion_inicial = toolbox.population(n=200)
    num_generaciones = 400
    prob_cruce = 0.9
    prob_mutacion = 0.2

    # Ejecutar el algoritmo genético
    resultados = algorithms.eaSimple(
        poblacion_inicial,
        toolbox,
        cxpb=prob_cruce,
        mutpb=prob_mutacion,
        ngen=num_generaciones,
        verbose=True
    )

    mejor_solucion = tools.selBest(poblacion_inicial, k=1)[0]
    ruta_mejor = [id_a_ciudad[i] for i in mejor_solucion]
    print("\nRuta más eficiente hallada:")
    print(" -> ".join(ruta_mejor))
    print("Distancia total:", calcular_distancia_total(mejor_solucion))

    dibujar_ruta_optima(mejor_solucion)

if __name__ == "__main__":
    ejecutar_algoritmo()
