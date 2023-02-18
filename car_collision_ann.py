import numpy as np

#En este proyecto, se ocupará una red neuronal artificial para predecir si ocurrirá una colisión a partir de 
#un conjunto de datos de entrada. Los datos de entrada serán un conjunto de atributos y la salida será si la
#colisión ocurrió o no.

# Fase A: Configuración

# 1.Definir la arquitectura de la red neural: Se definen el número de nodos de entrada, los nodos de salida así como
# el número de capas ocultas y los nodos que la conforman. Para este proyecto utilizaremos 4 nodos de entrada los cuales
# representan las caracteristicas a analizar de nuestro conjunto de datos, además, utilizaremos un nodo de salida pues solo 
# nos interesa conocer si se producirá una colisión o no. Tambien utilizaremos 5 nodos para la capa oculta, 
# utilizaremos este número pues se sugiere utilizar el número de nodos de entrada mas el número de nodos de salida como primer
# acercamiento para diseñar la red neural. 

# 2. Inicializar los pesos de cada neurona: Los pesos de la red neural deben ser inicializados con un valor cualquiera, en 
# nuestra implementación utilizaremos valores aleatorios para nuestros pesos iniciales y a partir de estos iremos ajustando 
# hasta que el porcentaje de error sea mínimo. 

# Fase B: Propagación hacia adelante:

#Para realizar la propagación hacia adelante se realizan los siguientes pasos:
# 1. Se multiplican los datos de entrada por su peso correspondiente.
# 2. Se suma el resultado de la multiplicación anterior para cada nodo oculto. 
# 3. Se aplica la función de activación al resultado de la suma y se envia este resultado al siguiente
# nodo oculto o al nodo salida.
# 4. Se repiten los pasos anteriores en el nodo de salida para conocer el resultado de nuestra predicción. 

# Fase C: Entrenamiento:
# 1. Se calcula el costo o error de nuestra predicción: Este costo es la diferencia entre el valor que calculado por la
# red neural y el valor esperado definido por la etiqueta de nuestros datos. El costo determina que tan buena o mala es 
# nuestra red neural. 

# 2. Ajuste de pesos en la red neural: Los pesos de cada nodo son lo único que podemos ajustar para modificar el comportamiento
# de nuestra red. Los pesos son esencialmente la inteligencia de nuestra red y deben ser ajustados para disminuir el costo. 

# 3. Condición de salida: El entrenamiento de la red no puede continuar de manera indefinida pues corremos el riesgo de que no le sea
# posible realizar predicciones fuera de los datos con los que fue entrenada. En este proyecto
# realizaremos el entrenamiento 1500 veces pues se trata de un conjunto de datos muy pequeño. 

# La función scale_dataset nos permite escar los datos de entrada a valorres entre 0 y 1 que nos permitirán 
# compararlos de manera eficiente y asignarle pesos adecuados para medir su contribución. 
# Es necesario hacer esto pues estamos comparando datos que no son medidos en la misma escala y es por ello 
# que son de distinta magnitud. Si no normalizamos los datos de entrada su contribución a la red no es la misma
# lo cual ocacionaría un sesgo hacia ese dato.
def scale_dataset(dataset, feature_count, feature_min, feature_max):
    scaled_data = []
    for data in dataset:
        example = []
        for i in range(0, feature_count):
            example.append(scale_data_feature(data[i], feature_min[i], feature_max[i]))
        scaled_data.append(example)
    return np.array(scaled_data)


# Para escalar nuestros datos utilizaremos el método de normalización min-max
def scale_data_feature(data, feature_min, feature_max):
    return (data - feature_min) / (feature_max - feature_min)

#La función sigmoid se usará como método de activación, para determinar si los nodos son activados
#Esta función se utiliza en el proceso de forward propagation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Parte de la función sigmoid y es utilizada en el proceso de backpropagation
def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))
    
#Clase para representar la red neuronal 
#En este caso se tiene una red con una sola capa
class NeuralNetwork:
    def __init__(self, features, labels, hidden_node_count):
        # Asigna las entradas, en este caso los 4 atributos de nuestro conjunto de datos
        self.input = features
        # Se hace una ponderación aleatoria para el peso entre las entradas y la capa
        self.weights_input = np.random.rand(self.input.shape[1], hidden_node_count)
        print(self.weights_input)
        # En un principio la salida de datos se inicializa en cero
        self.hidden = None
        # Se hace una ponderación aleatoria para el peso entre la capa y la salida
        self.weights_hidden = np.random.rand(hidden_node_count, 1)
        print(self.weights_hidden)
        # Salida correspondiente a la predicción de colisión
        self.expected_output = labels
        # Se inicializa la salida de datos en cero
        self.output = np.zeros(self.expected_output.shape)

    #Se agregan los casos con los atributos del conjunto de datos
    def add_example(self, features, label):
        np.append(self.input, features)
        np.append(self.expected_output, label)

#Proceso de propagación hacia adelante realiza la multiplicación entre valores escalados de entrada y 
#los pesos de cada nodo, esto también nos sirve en la salida realizando la misma función.
#Este proceso nos sirve para tener un punto de partida que posteriormente en el backpropagation se resolverá.
#Para poder obtener una prediccion, se utilizarán los pesos y la ponderación aleatoria nodos ocultos.
    def forward_propagation(self):
        #Se  suman los resultados resultado de la multiplicación entre pesos y las entradas
        hidden_weighted_sum = np.dot(self.input, self.weights_input)
        #Se activan los nodos respectivos
        self.hidden = sigmoid(hidden_weighted_sum)
        #Se  suman los resultados resultado de la multiplicación entre pesos y las salidas
        output_weighted_sum = np.dot(self.hidden, self.weights_hidden)  
        # Se obtiene la predicción inicial.
        self.output = sigmoid(output_weighted_sum)

    #Processo de backpropagation donde se calcula costo y actualizan los pesos
    #por tal motivo es necesario actualizar las ponderaciones mediante
    #la regla de la cadena y agrega los resultados de los datos actualizado a los pesos actuales.
    def back_propagation(self):
        cost = self.expected_output - self.output
        # print('ACTUAL: ')
        # print(self.expected_output)
        # print('PREDICTED: ')
        # print(self.output)
        # print('COSTS: ')
        # print(cost)
        # print('HIDDEN: ')
        # print(self.hidden)
        weights_hidden_update = np.dot(self.hidden.T, (2 * cost * sigmoid_derivative(self.output)))
        # print('WEIGHTS HIDDEN UPDATE:')
        # print(weights_hidden_update)
        weights_input_update = np.dot(self.input.T, (np.dot(2 * cost * sigmoid_derivative(self.output), self.weights_hidden.T) * sigmoid_derivative(self.hidden)))
        # print('WEIGHTS INPUT UPDATE:')
        # print(weights_hidden_update)

        # update the weights with the derivative (slope) of the loss function
        self.weights_hidden += weights_hidden_update
        # print('WEIGHTS HIDDEN:')
        # print(weights_hidden_update)

        self.weights_input += weights_input_update
        # print('WEIGHTS INPUT:')
        # print(weights_hidden_update)


#Proceso para correr la red neuronal. Acepta 'epochs' o número de iteraciones (para entrenar) como entrada al igual que 
#los datos escalados (min-max), las etiquetas y el número de nodos ocultos. 
#Esta función escala los datos y crea una nueva red neuronal con los datos de entrada prporcionados.
def run_neural_network(feature_data, label_data, feature_count, features_min, features_max, hidden_node_count, epochs):
    # Se aplica la escala mínima-máxima al conjunto de datos
    scaled_feature_data = scale_dataset(feature_data, feature_count, features_min, features_max)
    # nn va a inicializar la red neuronal con los datos escalados y los nodos ocultos proporcionados
    nn = NeuralNetwork(scaled_feature_data, label_data, hidden_node_count)
    # Se utiliza un for para entrenar la red neuronal en muchas iteraciones con los mismos datos de entrenamiento, 
    # mandando a llamar las propagaciones para calcular distintos datos como lo es el costo por ejemplo 
    for epoch in range(epochs):
        nn.forward_propagation()
        nn.back_propagation()
    # El siguiente for imprime la salida obtenida de la red neuronal
    print('OUTPUTS: ')
    for r in nn.output:
        print(r)
    # Se imprimen los pesos de entrada
    print('INPUT WEIGHTS: ')
    print(nn.weights_input)
    # Se imprimen los pesos ocultos
    print('HIDDEN WEIGHTS: ')
    print(nn.weights_hidden)

# Si se cumple la siguiente condición se ejecutara el programa
if __name__ == '__main__':
    # Número de atributos en el conjunto de datos
    FEATURE_COUNT = 4
    # Valores mínimos posibles para los atributos (velocidad, calidad del terreno, ángulo de visión, experiencia de conducción)
    FEATURE_MIN = [0, 0, 0, 0]
    # Valores máximos posibles para los atributos (velocidad, calidad del terreno, ángulo de visión, experiencia de conducción)
    FEATURE_MAX = [120, 10, 360, 400000]
    # Número de nodos ocultos
    HIDDEN_NODE_COUNT = 5
    # Número de iteraciones para entrenar la red neuronal
    EPOCHS = 1500

    # Datos de una colision en un arreglo en el orden de (velocidad, calidad del terreno, ángulo de visión, experiencia de conducción)
    #Este conjunto de datos representa a 7 conductores diferentes con valores de atributos diferentes para entrenar la red 
    car_collision_data = np.array([
        [65, 5,	180, 80000],
        [120, 1, 72, 110000],
        [8,	6,	288, 50000],
        [50, 2,	324, 1600],
        [25, 9,	36, 160000],
        [80, 3,	120, 6000],
        [40, 3,	360, 400000]
    ])

    # Etiquetas para el conjunto de datos de entrenamiento en un arreglo 
    #(0 = No se produjo ninguna colisión; 1 = Sí se produjo una colisión)
    car_collision_data_labels = np.array([
        [0],
        [1],
        [0],
        [1],
        [0],
        [1],
        [0]])

    # Finalmente se corre la red neuronal con todos los datos proporcionados
    run_neural_network(car_collision_data,
                       car_collision_data_labels,
                       FEATURE_COUNT,
                       FEATURE_MIN,
                       FEATURE_MAX,
                       HIDDEN_NODE_COUNT,
                       EPOCHS)
    
    
