import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
import copy
# Load data
X = pd.read_csv('Datasets/X.csv').drop(columns=['Unnamed: 0']).values
y = pd.read_csv('Datasets/y.csv').drop(columns=['Unnamed: 0']).rename(columns=({'0':'label'})).values.flatten()

X_non_linear = pd.read_csv('Datasets/Xnonlinear.csv').drop(columns=['Unnamed: 0']).values
Y_non_linear = pd.read_csv('Datasets/ynonlinear.csv').drop(columns=['Unnamed: 0']).rename(columns=({'0':'label'})).values.flatten()

XXOR = pd.read_csv('Datasets/XXOR.csv').values
YXOR = pd.read_csv('Datasets/YXOR.csv').rename(columns=({'Output':'label'})).values.flatten()
print(X_non_linear.shape)
print(Y_non_linear.shape)
# Plot data

def plot_data(X, y, title=""):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.grid(True)
    plt.show()

plot_data(X, y, title="Datos linealmente separables")
plot_data(X_non_linear, Y_non_linear, title="Datos no linealmente separables")
plot_data(XXOR, YXOR, title="Datos XOR")

def sequential_perceptron(x, y, epochs, learning_rate):
    weights = np.random.rand(x.shape[1])
    bias = np.random.rand()
    error_log = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(x.shape[0]):
            y_hat = np.dot(weights, x[i]) + bias
            prediction = 1 if y_hat >= 0 else 0

            if y[i] - prediction != 0:
                weights += learning_rate * (y[i] - prediction) * x[i]
                bias += learning_rate * (y[i] - prediction)
                total_error += np.abs(y[i] - prediction)
        if isinstance(total_error, np.ndarray):
            total_error = total_error.item()
        error_log.append(total_error)
    return weights, bias, error_log


def batch_perceptron(x, y, epochs, learning_rate):
    weights = np.random.rand(x.shape[1])
    bias = np.random.rand()
    error_log = []  
    for epoch in range(epochs):
        weights_update = np.zeros(x.shape[1])
        bias_update = 0
        total_error = 0

        for i in range(x.shape[0]):
            y_hat = np.dot(weights, x[i]) + bias
            prediction = 1 if y_hat >= 0 else 0

            if y[i] - prediction != 0:
                weights_update += learning_rate * (y[i] - prediction) * x[i]
                bias_update += learning_rate * (y[i] - prediction)
                total_error += np.abs(y[i] - prediction)
        if isinstance(total_error, np.ndarray):
            total_error = total_error.item()
        error_log.append(total_error)
        weights += weights_update
        bias += bias_update

        predictions = np.dot(x, weights) + bias
        predictions = np.where(predictions >= 0, 1, 0)
    return weights, bias, error_log

def plot_decision_boundary(X, y, weights, bias, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights[:2]) + bias
    Z = np.where(Z >= 0, 1, 0)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.show()
def plot_accuracy(error_log, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(error_log)), error_log)
    plt.title(title)
    plt.xlabel("Iteraciones")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()


# Algoritmo Genético Adaptado
def genetic_algorithm_perceptron(X, y, population_size=100, generations=100, mutation_rate=0.01, crossover_rate=0.7, mutation_step_size=0.1):
    num_features = X.shape[1]
    chromosome_length = num_features + 1 
    
    population = [np.random.uniform(-1, 1, chromosome_length) for _ in range(population_size)]
    best_errors = []
    best_individual = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitness_scores = []

        for individual in population:
            weights = individual[:-1]
            bias = individual[-1]
            activation = np.dot(X, weights) + bias
            predictions = np.where(activation >= 0, 1, 0)
            errors = np.sum(predictions != y)
            fitness_scores.append(errors)

            # Actualizar el mejor individuo
            if errors < best_fitness:
                best_fitness = errors
                best_individual = individual.copy()

        best_errors.append(best_fitness)

        # Selección de los 30% mejores individuos
        selected = []
        for _ in range(int(population_size * 0.3)):
            idx = np.argmin(fitness_scores)
            selected.append(population[idx])
            fitness_scores.pop(idx)

        # Cruce entre los individuos seleccionados
        newpob = []
        while len(newpob) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            if random.random() < crossover_rate:
                crossover_point = random.randint(1, chromosome_length - 1)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                newpob.append(child1)
                newpob.append(child2)

        # Mutación
        for individual in newpob:
            for gene_index in range(chromosome_length):
                if random.random() < mutation_rate:
                    individual[gene_index] += np.random.normal(0, mutation_step_size)
                    # Limitar los valores opcionalmente
                    individual[gene_index] = np.clip(individual[gene_index], -10, 10)

        # Reemplazar la población antigua con la nueva descendencia
        population = newpob



    # Devolver los mejores pesos, sesgo y registro de errores
    best_weights = best_individual[:-1]
    best_bias = best_individual[-1]
    return best_weights, best_bias, best_errors


# Inicializar la población
def initialize_population(lb, ub, dim, num_wolves):
    population = [lb + (ub - lb) * np.random.rand(dim) for _ in range(num_wolves)]
    return population

# Función de fitness
def fitness_function(weights_bias, X, y):
    weights = weights_bias[:-1]
    bias = weights_bias[-1]
    predictions = np.dot(X, weights) + bias
    predictions = np.where(predictions >= 0, 1, 0)
    error = np.sum(predictions != y)  
    return error

# Ordenar población por fitness
def sort_population(population, X, y):
    return sorted(population, key=lambda wolf: fitness_function(wolf, X, y))

# Actualizar posiciones de los lobos
def update_positions(alpha, beta, delta, population, a, lb, ub):
    new_population = []
    for wolf in population:
        new_wolf = np.zeros_like(wolf)
        for j in range(len(wolf)):
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[j] - wolf[j])
            X1 = alpha[j] - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[j] - wolf[j])
            X2 = beta[j] - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[j] - wolf[j])
            X3 = delta[j] - A3 * D_delta

            new_wolf[j] = (X1 + X2 + X3) / 3
            new_wolf[j] = np.clip(new_wolf[j], lb[j], ub[j])

        new_population.append(new_wolf)
    return new_population

# Algoritmo GWO para entrenar el perceptrón
def GWO_perceptron(X, y, lb, ub, num_wolves=30, epochs=100):
    dim = X.shape[1] + 1  # Dimensión: número de características + 1 (bias)
    population = initialize_population(lb, ub, dim, num_wolves)
    population = sort_population(population, X, y)

    alpha = population[0]
    beta = population[1]
    delta = population[2]

    a = 2
    error_log = []

    for epoch in range(epochs):
        a = 2 - epoch * (2 / epochs)
        population = update_positions(alpha, beta, delta, population, a, lb, ub)
        population = sort_population(population, X, y)

        alpha = population[0]
        beta = population[1]
        delta = population[2]

        current_error = fitness_function(alpha, X, y)
        error_log.append(current_error)


    return alpha, error_log


# Perceptrón secuencial
seq_data = [[], [],[]]
seq_data = sequential_perceptron(X, y, epochs=100, learning_rate=0.1)

# Perceptrón por lotes
batch_data = [[], [],[]]
batch_data = batch_perceptron(X, y, epochs=120, learning_rate=0.1)

# GWO
gwo_data = [[], [],[]]
lb = np.array([-5] * (X.shape[1] + 1))  # Límite inferior
ub = np.array([5] * (X.shape[1] + 1))   # Límite superior
gwo_data = GWO_perceptron(X, y, lb, ub, num_wolves=50, epochs=50)

# Entrenar el perceptrón usando el algoritmo genético
ga_data = genetic_algorithm_perceptron(X, y, population_size=1000, generations=100, mutation_rate=0.03, crossover_rate=0.7, mutation_step_size=0.1)


# Gráficas
plot_decision_boundary(X, y, seq_data[0], seq_data[1], "Perceptrón secuencial - Datos linealmente separables")
plot_accuracy(seq_data[2], "Perceptrón secuencial - Datos linealmente separables")

plot_decision_boundary(X, y, batch_data[0], batch_data[1], "Perceptrón por lotes - Datos linealmente separables")
plot_accuracy(batch_data[2], "Perceptrón por lotes - Datos linealmente separables")

plot_decision_boundary(X, y, ga_data[0], ga_data[1], "Algoritmo Genético - Datos linealmente separables")
plot_accuracy(ga_data[2], "Algoritmo Genético - Evolución del Error")

plot_decision_boundary(X, y, gwo_data[0][:-1], gwo_data[0][-1], "GWO - Datos linealmente separables")
plot_accuracy(gwo_data[1], "GWO - Datos linealmente separables")

X_non_linear = pd.read_csv('Datasets/Xnonlinear.csv').drop(columns=['Unnamed: 0']).values
Y_non_linear = pd.read_csv('Datasets/ynonlinear.csv').drop(columns=['Unnamed: 0']).rename(columns=({'0':'label'})).values.flatten()
#Cambiar los -1 por 0 
Y_non_linear = np.where(Y_non_linear == -1, 0, Y_non_linear)
print(X_non_linear.shape)
print(Y_non_linear.shape)

third_column = X_non_linear[:, 0] ** 2 + X_non_linear[:, 1] ** 2
X_non_linear = np.column_stack((X_non_linear, third_column))


# Graficar los datasets en 3D

def plot_3d_dataset(X, y, title=""):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)
    ax.set_title(title)
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_zlabel('Característica 3')
    plt.show()
# Función para graficar en 3D
def plot_3d_decision_boundary(X, y, weights, bias, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar puntos de datos
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)

    # Crear una malla para la superficie de decisión
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                         np.linspace(y_min, y_max, 20))
    zz = (-weights[0]*xx - weights[1]*yy - bias) / weights[2]

    ax.plot_surface(xx, yy, zz, alpha=0.5, color='yellow')
    ax.set_title(title)
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_zlabel('Característica 3')
    plt.show()
plot_3d_dataset(X_non_linear, Y_non_linear, title="Datos no linealmente separables")

# Perceptrón secuencial
seq_data = [[], [],[]]
seq_data = sequential_perceptron(X_non_linear, Y_non_linear, epochs=100    , learning_rate=0.1)
print(batch_data[0])

plot_3d_decision_boundary(X_non_linear, Y_non_linear, seq_data[0], seq_data[1], "Perceptrón secuencial - Datos no linealmente separables")
plot_accuracy(seq_data[2], "Perceptrón secuencial - Datos no linealmente separables")

batch_data = [[], [],[]]
batch_data = batch_perceptron(X_non_linear, Y_non_linear, epochs=1000, learning_rate=0.1)

plot_3d_decision_boundary(X_non_linear, Y_non_linear, batch_data[0], batch_data[1], "Perceptrón por lotes - Datos no linealmente separables")
plot_accuracy(batch_data[2], "Perceptrón por lotes - Datos no linealmente separables")

ga_data = genetic_algorithm_perceptron(X_non_linear, Y_non_linear, population_size=100, generations=100, mutation_rate=0.01, crossover_rate=0.7, mutation_step_size=0.1)

plot_3d_decision_boundary(X_non_linear, Y_non_linear, ga_data[0], ga_data[1], "Algoritmo Genético - Datos no linealmente separables")
plot_accuracy(ga_data[2], "Algoritmo Genético - Datos no linealmente separables")

gwo_data = [[], [],[]]
lb = np.array([-5] * (X_non_linear.shape[1] + 1))  # Límite inferior
ub = np.array([5] * (X_non_linear.shape[1] + 1))   # Límite superior
gwo_data = GWO_perceptron(X_non_linear, Y_non_linear, lb, ub, num_wolves=30, epochs=100)
plot_3d_decision_boundary(X_non_linear, Y_non_linear, gwo_data[0][:-1], gwo_data[0][-1], "GWO - Datos no linealmente separables")
plot_accuracy(gwo_data[1], "GWO - Datos no linealmente separables")

#Transformar XOR
XXOR = pd.read_csv('Datasets/XXOR.csv').values
YXOR = pd.read_csv('Datasets/YXOR.csv').rename(columns=({'Output':'label'})).values.flatten()
third_column = XXOR[:, 0] * XXOR[:, 1] 
XXOR = np.column_stack((XXOR, third_column))

plot_3d_dataset(XXOR, YXOR, title="Datos XOR")

# Perceptrón secuencial
seq_data = [[], [],[]]
seq_data = sequential_perceptron(XXOR, YXOR, epochs=100, learning_rate=0.1)

# Percetrón por lotes
batch_data = [[], [],[]]
batch_data = batch_perceptron(XXOR, YXOR, epochs=100, learning_rate=0.1)

# GA
ga_data = genetic_algorithm_perceptron(XXOR, YXOR, population_size=100, generations=100, mutation_rate=0.01, crossover_rate=0.7, mutation_step_size=0.1)

# GWO
gwo_data = [[], [],[]]
lb = np.array([-5] * (XXOR.shape[1] + 1))  # Límite inferior
ub = np.array([5] * (XXOR.shape[1] + 1))   # Límite superior
gwo_data = GWO_perceptron(XXOR, YXOR, lb, ub, num_wolves=50, epochs=100)

# Gráficas

plot_3d_decision_boundary(XXOR, YXOR, seq_data[0], seq_data[1], "Perceptrón secuencial - Datos XOR")
plot_accuracy(seq_data[2], "Perceptrón secuencial - Datos XOR")

plot_3d_decision_boundary(XXOR, YXOR, batch_data[0], batch_data[1], "Perceptrón por lotes - Datos XOR")
plot_accuracy(batch_data[2], "Perceptrón por lotes - Datos XOR")

plot_3d_decision_boundary(XXOR, YXOR, ga_data[0], ga_data[1], "Algoritmo Genético - Datos XOR")
plot_accuracy(ga_data[2], "Algoritmo Genético - Datos XOR")

plot_3d_decision_boundary(XXOR, YXOR, gwo_data[0][:-1], gwo_data[0][-1], "GWO - Datos XOR")
plot_accuracy(gwo_data[1], "GWO - Datos XOR")
