import random
import math

def calcular_conflictos(tablero):
    """
    Calcula el número de ataques entre reinas en el tablero.
    """
    n = len(tablero)
    conflictos = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if abs(tablero[i] - tablero[j]) == abs(i - j):  # Diagonales
                conflictos += 1
    
    return conflictos

def generar_vecino(tablero):
    """
    Genera un vecino intercambiando dos reinas al azar.
    """
    n = len(tablero)
    nuevo_tablero = tablero[:]
    i, j = random.sample(range(n), 2)
    nuevo_tablero[i], nuevo_tablero[j] = nuevo_tablero[j], nuevo_tablero[i]
    return nuevo_tablero, (i, j)

def recocido_simulado(tablero_inicial, temperatura=1000, factor_enfriamiento=0.99, iteraciones=1000):
    """
    Algoritmo de búsqueda de Recocido Simulado para el problema de las N-Reinas.
    """
    tablero_actual = tablero_inicial[:]
    mejor_solucion = tablero_actual[:]
    conflictos_actual = calcular_conflictos(tablero_actual)
    mejor_costo = conflictos_actual
    temperatura_actual = temperatura
    
    for i in range(iteraciones):
        if conflictos_actual == 0:
            break  # Solución encontrada
        
        tablero_nuevo, movimiento = generar_vecino(tablero_actual)
        conflictos_nuevo = calcular_conflictos(tablero_nuevo)
        
        delta = conflictos_nuevo - conflictos_actual
        if delta < 0 or random.random() < math.exp(-delta / temperatura_actual):
            tablero_actual = tablero_nuevo
            conflictos_actual = conflictos_nuevo
            if conflictos_actual < mejor_costo:
                mejor_solucion = tablero_actual[:]
                mejor_costo = conflictos_actual
        
        temperatura_actual *= factor_enfriamiento
        
        # Mostrar información detallada de la iteración
        print(f"\nIteración {i+1}")
        print(f"Solución actual: {tablero_actual}")
        print(f"Conflictos actuales: {conflictos_actual}")
        print(f"Mejor solución encontrada: {mejor_solucion}")
        print(f"Mejor costo: {mejor_costo}")
        print(f"Temperatura actual: {temperatura_actual:.4f}")
    
    return mejor_solucion, mejor_costo

def main():
    """
    Función principal para ingresar datos y ejecutar el algoritmo.
    """
    n = int(input("Ingrese el tamaño del tablero (N): "))
    print("Ingrese el orden de las reinas en cada columna (separado por espacios):")
    tablero_inicial = list(map(int, input().split()))
    
    if len(tablero_inicial) != n:
        print("Error: La cantidad de valores ingresados no coincide con el tamaño del tablero.")
        return
    
    solucion, conflictos = recocido_simulado(tablero_inicial)
    
    print("\nTablero final:", solucion)
    print("Número de conflictos:", conflictos)
    
    if conflictos == 0:
        print("¡Se encontró una solución válida!")
    else:
        print("No se encontró una solución óptima, intente nuevamente.")

if __name__ == "__main__":
    main()

