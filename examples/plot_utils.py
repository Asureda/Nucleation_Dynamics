import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Configuraciones de la paleta de colores y la estética de la gráfica
cmap = cm.get_cmap("viridis")

# Supongamos que 'num_points' es el número de diferentes tiempos que quieres mostrar
num_points = 20
# Supongamos que 'indices' selecciona momentos específicos en 'sim.time'
indices = np.linspace(2, len(sim.time) - 1, num_points, dtype=int)

# Creación de la figura y ajuste de tamaño para alta calidad de impresión
fig, ax = plt.subplots(figsize=(6, 12), dpi=300)
vertical_spacing = 2  # Espaciado vertical entre cada gráfica

for i, index in enumerate(indices):
    # Simula obtener datos de densidad (reemplaza con tus propios datos)
    Y = sim.cluster_array[:, index] / cluster_physics.number_density_equilibrium(number_molecules_array).magnitude
    X = number_molecules_array

    # Aumenta el desplazamiento para cada serie de tiempo
    offset = i * vertical_spacing
    ax.plot(X, Y + offset, color="black", linestyle='-', linewidth=1.1, zorder=100 - i)
    color = cmap(float(i) / len(indices))  # Usa el índice para variar el color

    # Rellenar bajo la curva con transparencia (alpha)
    ax.fill_between(X, Y + offset, offset, color=color, alpha=0.5, zorder=50 - i)

# Configuraciones de los ejes y la estética
ax.yaxis.set_tick_params(tick1On=False)  # Desactiva ticks del eje y
ax.set_xlim(1, 41)  # Ajusta según tu rango de número de moléculas
ax.set_ylim(-1, num_points * vertical_spacing)  # Ajusta según la cantidad de series y el espaciado

# Línea vertical de referencia
ax.axvline(0.0, ls="--", lw=0.75, color="black", zorder=50)

# Configuración de las etiquetas del eje y
ax.yaxis.set_tick_params(labelleft=True)
ax.set_yticks([i * vertical_spacing for i in range(num_points)])
ax.set_yticklabels(["Tiempo %d" % (i + 1) for i in range(num_points)], fontsize=8)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_verticalalignment("bottom")

# Etiquetas de los ejes
ax.set_xlabel('Número de Moléculas', fontsize=10)
ax.set_ylabel('Densidad (ajustada para cada tiempo)', fontsize=10)

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D  # Importa la herramienta necesaria para gráficos 3D

num_points = 200
# Definir el rango de tiempos y números de moléculas para el gráfico de contorno
number_molecules_array = np.arange(1, MAX_NUMBER_MOLECULES + 1)
time_array = np.linspace(sim2.time[1], sim2.time.max(), num_points)  # Asume que sim.time contiene los tiempos
density_matrix = np.zeros((len(number_molecules_array), num_points))  # Matriz para almacenar los datos

# Calcular los datos de densidad para cada combinación de tiempo y número de moléculas
for t_index, time in enumerate(time_array):
    for n_index, num_mol in enumerate(number_molecules_array):
        # Calcula la densidad para este tiempo y número de moléculas y guárdala en la matriz
        # Nota: Ajusta la siguiente línea según tus cálculos específicos y nombres de variables
        density_matrix[n_index, t_index] = (sim2.cluster_array[num_mol-1, t_index] )

# Creación de la figura y los ejes para un gráfico 3D
fig = plt.figure(figsize=(10, 7), dpi=300)
ax = fig.add_subplot(111, projection='3d')  # Configura los ejes para gráfico 3D

# Creamos las mallas para el gráfico, basadas en el tiempo y número de moléculas
X, Y = np.meshgrid(time_array/np.max(time_array), number_molecules_array)  # Ajusta por tu tasa de salto si es necesario

# Convertimos la densidad en logaritmo para una mejor visualización y evitar problemas con valores muy grandes o muy pequeños
Z = np.log10(density_matrix)  # Asegúrate de que no hay valores negativos o cero antes de aplicar log10

# Crea la superficie 3D
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.8, antialiased=True)

contour = ax.contour(X, Y, Z, 100, cmap='viridis')  # 20 niveles de contorno y mapa de colores 'viridis'

# Añadir una barra de color que mapea los valores a colores
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)  # Ajusta estos valores según sea necesario
cbar.set_label('Log Density', fontsize=12)

# Ajustes finales de etiquetas, títulos y otros elementos gráficos
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Number of Molecules', fontsize=14)
ax.set_zlabel('Log Density', fontsize=14)
ax.set_title('Temporal Evolution of Number Density', fontsize=16)
ax.tick_params(labelsize=12)  # Ajusta el tamaño de los ticks si es necesario
ax.view_init(elev=30, azim=45)  # Elevación y azimut
plt.show()
