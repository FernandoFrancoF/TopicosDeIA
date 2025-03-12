import random

def contar_colisiones(solucion):
    n = len(solucion)
    colisiones = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(solucion[i] - solucion[j]) == abs(i - j):
                colisiones += 1
    return colisiones

def generar_vecindario(solucion):
    vecindario = []
    n = len(solucion)
    for i in range(n):
        for j in range(i + 1, n):
            vecino = solucion[:]
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecindario.append((vecino, contar_colisiones(vecino)))
    return sorted(vecindario, key=lambda x: x[1])

def busqueda_tabu(n, solucion_inicial, max_iter=100, tabu_tenure=3):
    solucion_actual = solucion_inicial[:]
    mejor_solucion = solucion_actual[:]
    mejor_colisiones = contar_colisiones(solucion_actual)
    lista_tabu = []
    
    for iteracion in range(max_iter):
        print(f"Iteración {iteracion + 1}: {solucion_actual}, Colisiones: {contar_colisiones(solucion_actual)}")
        vecindario = generar_vecindario(solucion_actual)
        
        for vecino, colisiones in vecindario:
            if vecino not in lista_tabu:
                solucion_actual = vecino[:]
                if colisiones < mejor_colisiones:
                    mejor_solucion = vecino[:]
                    mejor_colisiones = colisiones
                break
        
        lista_tabu.append(solucion_actual)
        if len(lista_tabu) > tabu_tenure:
            lista_tabu.pop(0)
        
        if mejor_colisiones == 0:
            break
    
    return mejor_solucion, mejor_colisiones

n = 8  # Número de reinas
solucion_inicial = list(map(int, input("Ingrese la configuración inicial separada por espacios: ").split()))
solucion, colisiones = busqueda_tabu(n, solucion_inicial)
print("Mejor solución encontrada:", solucion)
print("Colisiones:", colisiones)