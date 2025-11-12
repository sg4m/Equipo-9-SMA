"""
Simulacion multiagente de robots de limpieza reactivos con analisis experimental.
Este programa modela una habitacion en grid donde agentes de limpieza
reactivos se mueven aleatoriamente para limpiar celdas sucias.
Incluye funcionalidad para correr experimentos multiples y analizar resultados.

Autores:
-Luis Ángel Godínez González A01752310
-Santiago Gamborino Morales A01753159
Fecha: 07 de Noviembre de 2025
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import MultiGrid 


class Room:
    """
    Clase que representa una habitacion con celdas limpias y sucias.
    """

    def __init__(self, filas, columnas, porcentajeSucio):
        """
        Inicializa la habitacion con dimensiones especificadas.

        Parametros:
        filas -- Numero de filas del grid (M)
        columnas -- Numero de columnas del grid (N)
        porcentajeSucio -- Porcentaje de celdas inicialmente sucias (0.0 a 1.0)
        """
        self.filas = filas
        self.columnas = columnas
        self.grid = np.zeros((filas, columnas), dtype=int)

        totalCeldas = filas * columnas
        numCeldasSucias = int(totalCeldas * porcentajeSucio)

        posiciones = [(i, j) for i in range(filas) for j in range(columnas)]
        celdasSucias = random.sample(posiciones, numCeldasSucias)

        for fila, columna in celdasSucias:
            self.grid[fila][columna] = 1

    def estaSucia(self, fila, columna):
        """
        Verifica si una celda especifica esta sucia.

        Parametros:
        fila -- Indice de fila de la celda
        columna -- Indice de columna de la celda

        Retorna:
        True si la celda esta sucia (valor 1), False si esta limpia (valor 0)
        """
        return self.grid[fila][columna] == 1

    def limpiarCelda(self, fila, columna):
        """
        Limpia una celda especifica estableciendo su valor a 0.

        Parametros:
        fila -- Indice de fila de la celda a limpiar
        columna -- Indice de columna de la celda a limpiar
        """
        self.grid[fila][columna] = 0

    def todasLimpias(self):
        """
        Verifica si todas las celdas del grid estan limpias.

        Retorna:
        True si no hay celdas sucias, False en caso contrario
        """
        return np.sum(self.grid) == 0

    def contarCeldasSucias(self):
        """
        Cuenta el numero total de celdas sucias en el grid.

        Retorna:
        Numero entero de celdas con valor 1 (sucias)
        """
        return np.sum(self.grid)


class CleaningAgent(Agent):
    """
    Agente de limpieza reactivo que limpia celdas sucias y se mueve aleatoriamente.
    """

    def __init__(self, model):
        """
        Inicializa un agente de limpieza.

        Parametros:
        modelo -- Referencia al modelo de simulacion que contiene este agente
        """
        super().__init__(model)
        self.movimientosExitosos = 0
        self.movimientosFallidos = 0
        self.celdasLimpiadas = 0

    def step(self):
        """
        Ejecuta un paso de simulacion del agente.
        El agente limpia si la celda actual esta sucia, o se mueve aleatoriamente
        a una celda vecina en la vecindad de Moore (8 direcciones).
        """
        filaActual = self.pos[0]
        columnaActual = self.pos[1]

        if self.model.habitacion.estaSucia(filaActual, columnaActual):
            self.model.habitacion.limpiarCelda(filaActual, columnaActual)
            self.celdasLimpiadas = self.celdasLimpiadas + 1
        else:
            vecinos = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False
            )

            if len(vecinos) > 0:
                nuevaPosicion = random.choice(vecinos)

                agentesEnDestino = self.model.grid.get_cell_list_contents([nuevaPosicion])

                filaDestino = nuevaPosicion[0]
                columnaDestino = nuevaPosicion[1]
                dentroDeLimites = (0 <= filaDestino < self.model.grid.height and 
                                   0 <= columnaDestino < self.model.grid.width)

                if dentroDeLimites and len(agentesEnDestino) == 0:
                    self.model.grid.move_agent(self, nuevaPosicion)
                    self.movimientosExitosos = self.movimientosExitosos + 1
                else:
                    self.movimientosFallidos = self.movimientosFallidos + 1


class CleaningModel(Model):
    """
    Modelo de simulacion multiagente para robots de limpieza reactivos.
    """

    def __init__(self, numAgentes, filas, columnas, porcentajeSucio, maxPasos, seed=None):
        """
        Inicializa el modelo de simulacion.

        Parametros:
        numAgentes -- Numero de agentes de limpieza (1 a 5)
        filas -- Numero de filas del grid (M)
        columnas -- Numero de columnas del grid (N)
        porcentajeSucio -- Porcentaje de celdas inicialmente sucias (0.0 a 1.0)
        maxPasos -- Numero maximo de pasos de simulacion
        seed -- Semilla para el generador de numeros aleatorios (opcional)
        """
        super().__init__(seed=seed)
        self.numAgentes = numAgentes
        self.habitacion = Room(filas, columnas, porcentajeSucio)
        self.grid = MultiGrid(columnas, filas, False)
        self.pasosActuales = 0
        self.maxPasos = maxPasos
        self.simulacionActiva = True

        # Crear agentes y colocarlos en posicion inicial
        for i in range(numAgentes):
            agente = CleaningAgent(self)
            self.grid.place_agent(agente, (0, 0))

    def step(self):
        """
        Ejecuta un paso de simulacion para todos los agentes.
        Verifica si la simulacion debe detenerse al completar la limpieza
        o alcanzar el numero maximo de pasos.
        """
        self.agents.shuffle_do("step")
        self.pasosActuales = self.pasosActuales + 1

        if self.habitacion.todasLimpias():
            self.simulacionActiva = False

        if self.pasosActuales >= self.maxPasos:
            self.simulacionActiva = False


def runSimulation(modelo):
    """
    Ejecuta la simulacion completa y recolecta metricas.

    Parametros:
    modelo -- Instancia de CleaningModel a ejecutar

    Retorna:
    Diccionario con metricas: tiempoPasos, porcentajeLimpio, movimientosTotales
    """
    while modelo.simulacionActiva:
        modelo.step()

    totalCeldas = modelo.habitacion.filas * modelo.habitacion.columnas
    celdasSucias = modelo.habitacion.contarCeldasSucias()
    celdasLimpias = totalCeldas - celdasSucias
    porcentajeLimpio = (celdasLimpias / totalCeldas) * 100.0

    movimientosTotales = 0
    for agente in modelo.agents:
        movimientosTotales = movimientosTotales + agente.movimientosExitosos

    resultados = {
        'tiempoPasos': modelo.pasosActuales,
        'porcentajeLimpio': porcentajeLimpio,
        'movimientosTotales': movimientosTotales
    }

    return resultados


def experiment_results(num_agents_list=[1, 2, 3, 4, 5], reps=10, M=10, N=10, 
                      dirty_percentage=0.3, max_steps=1000):
    """
    Ejecuta experimentos multiples variando el numero de agentes.
    Recolecta metricas estadisticas de multiples repeticiones.

    Parametros:
    num_agents_list -- Lista con numeros de agentes a probar
    reps -- Numero de repeticiones por configuracion
    M -- Numero de filas del grid
    N -- Numero de columnas del grid
    dirty_percentage -- Porcentaje de celdas inicialmente sucias
    max_steps -- Numero maximo de pasos por simulacion

    Retorna:
    DataFrame de pandas con promedios y desviaciones estandar por configuracion
    """
    print("=" * 70)
    print("INICIANDO EXPERIMENTOS DE LIMPIEZA MULTIAGENTE")
    print("=" * 70)
    print(f"Configuracion del grid: {M}x{N}")
    print(f"Porcentaje de celdas sucias: {dirty_percentage * 100}%")
    print(f"Repeticiones por configuracion: {reps}")
    print(f"Numero maximo de pasos: {max_steps}")
    print("=" * 70)

    # Lista para almacenar datos de todas las corridas
    datosExperimento = []

    # Iterar sobre cada numero de agentes
    for numAgentes in num_agents_list:
        print(f"\nProbando con {numAgentes} agente(s)...")

        # Ejecutar multiples repeticiones
        for rep in range(reps):
            # Crear modelo con seed unica para reproducibilidad
            seed = numAgentes * 1000 + rep
            modelo = CleaningModel(numAgentes, M, N, dirty_percentage, max_steps, seed=seed)

            # Ejecutar simulacion
            resultados = runSimulation(modelo)

            # Almacenar resultados
            datosExperimento.append({
                'num_agents': numAgentes,
                'repeticion': rep + 1,
                'seed': seed,
                'tiempo_pasos': resultados['tiempoPasos'],
                'porcentaje_limpio': resultados['porcentajeLimpio'],
                'movimientos_totales': resultados['movimientosTotales']
            })

            print(f"  Repeticion {rep + 1}/{reps} completada: "
                  f"{resultados['tiempoPasos']} pasos, "
                  f"{resultados['porcentajeLimpio']:.1f}% limpio")

    # Crear DataFrame con todos los datos
    dfCompleto = pd.DataFrame(datosExperimento)

    # Calcular estadisticas agregadas por numero de agentes
    estadisticas = dfCompleto.groupby('num_agents').agg({
        'tiempo_pasos': ['mean', 'std'],
        'porcentaje_limpio': ['mean', 'std'],
        'movimientos_totales': ['mean', 'std']
    }).reset_index()

    # Aplanar nombres de columnas
    estadisticas.columns = [
        'num_agents',
        'avg_time_to_clean',
        'std_time',
        'avg_clean_percentage',
        'std_clean_percentage',
        'avg_total_movements',
        'std_total_movements'
    ]

    print("\n" + "=" * 70)
    print("EXPERIMENTOS COMPLETADOS")
    print("=" * 70)

    return estadisticas, dfCompleto


def visualize_results(estadisticas):
    """
    Crea visualizaciones de los resultados experimentales.

    Parametros:
    estadisticas -- DataFrame con estadisticas agregadas por num_agents
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico 1: Tiempo promedio vs Numero de agentes
    axes[0].errorbar(
        estadisticas['num_agents'],
        estadisticas['avg_time_to_clean'],
        yerr=estadisticas['std_time'],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        color='#2E86AB',
        ecolor='#A23B72'
    )
    axes[0].set_xlabel('Numero de Agentes', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Tiempo Promedio (pasos)', fontsize=12, fontweight='bold')
    axes[0].set_title('Tiempo de Limpieza vs Numero de Agentes', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks(estadisticas['num_agents'])

    # Grafico 2: Movimientos totales promedio vs Numero de agentes
    axes[1].errorbar(
        estadisticas['num_agents'],
        estadisticas['avg_total_movements'],
        yerr=estadisticas['std_total_movements'],
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        color='#F18F01',
        ecolor='#C73E1D'
    )
    axes[1].set_xlabel('Numero de Agentes', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Movimientos Totales Promedio', fontsize=12, fontweight='bold')
    axes[1].set_title('Movimientos Totales vs Numero de Agentes', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(estadisticas['num_agents'])

    plt.tight_layout()
    plt.savefig('resultados_experimento_limpieza.png', dpi=300, bbox_inches='tight')
    print("\nGrafico guardado como 'resultados_experimento_limpieza.png'")
    plt.show()


def main():
    """
    Funcion principal para ejecutar experimentos completos.
    Configura parametros, ejecuta experimentos y genera visualizaciones.
    """
    # Parametros del experimento
    M = 10
    N = 10
    DIRTY_PERCENTAGE = 0.3
    NUM_AGENTS_LIST = [1, 2, 3, 4, 5]
    REPETICIONES = 10
    MAX_STEPS = 1000

    # Ejecutar experimentos
    estadisticas, dfCompleto = experiment_results(
        num_agents_list=NUM_AGENTS_LIST,
        reps=REPETICIONES,
        M=M,
        N=N,
        dirty_percentage=DIRTY_PERCENTAGE,
        max_steps=MAX_STEPS
    )

    # Mostrar tabla de resultados
    print("\n" + "=" * 70)
    print("TABLA DE RESULTADOS ESTADISTICOS")
    print("=" * 70)
    print(estadisticas.to_string(index=False))
    print("=" * 70)

    # Guardar resultados en CSV
    estadisticas.to_csv('estadisticas_experimento.csv', index=False)
    dfCompleto.to_csv('datos_completos_experimento.csv', index=False)
    print("\nDatos guardados en:")
    print("  - estadisticas_experimento.csv")
    print("  - datos_completos_experimento.csv")

    # Crear visualizaciones
    visualize_results(estadisticas)

    # Analisis adicional
    print("\n" + "=" * 70)
    print("ANALISIS ADICIONAL")
    print("=" * 70)
    
    mejorConfig = estadisticas.loc[estadisticas['avg_time_to_clean'].idxmin()]
    print(f"\nConfiguracion mas rapida:")
    print(f"  Numero de agentes: {int(mejorConfig['num_agents'])}")
    print(f"  Tiempo promedio: {mejorConfig['avg_time_to_clean']:.2f} pasos")
    print(f"  Desviacion estandar: {mejorConfig['std_time']:.2f} pasos")

    mejorEficiencia = estadisticas.loc[
        (estadisticas['avg_total_movements'] / estadisticas['avg_time_to_clean']).idxmin()
    ]
    print(f"\nConfiguracion mas eficiente (menos movimientos por paso):")
    print(f"  Numero de agentes: {int(mejorEficiencia['num_agents'])}")
    movimientosPorPaso = (mejorEficiencia['avg_total_movements'] / 
                          mejorEficiencia['avg_time_to_clean'])
    print(f"  Movimientos por paso: {movimientosPorPaso:.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
