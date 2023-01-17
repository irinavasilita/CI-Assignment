# Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import BallTree
from drawtree import draw_level_order


# The problem:
# There are 2 classes, red and green, and we want to separate them
# by drawing a straight line between them. 


# The perceptron algorithm
def perceptron(X, y, a, iterations):
    # X - the input matrix with the "1" column for the bias already inserted
    # y - the labels/targets
    # a - alpha, learning rate

    # m - the number of training examples
    # n - the number of features (in our case 2 features)
    m, n = X.shape
    y = np.array(y)

    #print("The targets array: ")
    #print(y)
    
    # Initializing theta (the parameters vector) with zeros
    theta = np.zeros((n,))
    #print("The initial weights array:")
    #print(theta)

    # Training
    for iteration in range(iterations):
        # Transpose of labels vector
        yT = y.T

        prediction = np.dot(theta, X.transpose())
        predicted_y = np.zeros(len(y))

        for i in range(len(y)):
            predicted_y[i] = unit_step(prediction[i])

        #print("The prediction (predicted_y):")
        #print(predicted_y)

        error = np.subtract(yT,predicted_y)
        #print("The error array: ")
        #print(error)

        if(sum(error) == 0):
            break
        
        # Update the weights
        theta = np.add(theta, a * (np.dot(error, X)))

    #print("Updated weights: ")
    #print(theta)

    return theta

def unit_step(z):
    return 1.0 if (z > 0) else 0.0

# The SVM algorithm
def SVM(X, y):
    svc_model = SVC(kernel='linear')
    svc_model.fit(X,y)

    theta = svc_model.coef_[0]           # theta consists of 2 elements
    return theta

# The Ball Tree algorithm
def DrawBalls(centroids, balls):
    colorList = ['r','b','g','y','m','c', '#eeefff', '#eeeffff']
    for i in range (0, len(balls)):
        circle = plt.Circle(centroids[i], balls[i][3], edgecolor=colorList[i], fill=False, label = f'Ball {i}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(circle)
        plt.legend()
        fig.canvas.draw()

def BallTreeAlg(points):
    X = np.array(points)
    #X = [[1,2], [2,6] , [3,5], [3,4], [5,6], [5,4], [7,7], [7,8], [8,3]]
    print(X)
    tree = BallTree(X, leaf_size=2) 
    tree_arrays = tree.get_arrays()
    print(tree_arrays)
    points_indexes = tree_arrays[1]
    #print(f'The ordered indexes are: {points_indexes}')
    centroids = tree_arrays[-1][0]
    #print(f'The centroids are: {centroids}')
    balls = tree_arrays[2]
    for i in range (0, len(balls)):
        str = f'The ball {i} with centroid in {centroids[i]} has the points with indexes {points_indexes[balls[i][0]:balls[i][1]]} and it has a radius of {balls[i][3]}. Is a leaf? {balls[i][2]}.'
        print(str)

    temp_arr = f'{list(range(0,len(balls)))}'
    draw_level_order(temp_arr)
    
    DrawBalls(centroids, balls)

    
# Initializing the input matrix
points = []
y = []

# Formatting the input
def format_input_X(points):
    X = np.array(points)
    # Insert a column with 1's for bias
    m = len(X)
    X0 = np.ones((m,1))
    X = np.insert(X, [0], X0, axis=1)

    #print(X)
    return X

# Define the figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

def plot_decision_boundary(theta, algorithm):
    # X --> Inputs
    # theta --> parameters

    x1 = np.array([-20, 20])
    print(type(x1))
    m = -theta[1]/theta[2]
    print(m)
    c = -theta[0]/theta[2]
    print(c)
    x2 = m*x1 + c
    print(x2)
    
    # Plotting
    if algorithm=='p':
        plt.plot(x1, x2, ls='-', c='r', label='Perceptron')
    if algorithm=='v':
        plt.plot(x1, x2, ls='-', c='g', label='Support vector machine')
    plt.legend()
    fig.canvas.draw()

# Events 
def onclick(event):
    if event.button == 1:
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            (event.button, event.x, event.y, event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        points.append([event.xdata, event.ydata])
        y.append((1))

    if event.button == 3:
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            (event.button, event.x, event.y, event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'gx')
        points.append([event.xdata, event.ydata])
        y.append((0))

    fig.canvas.draw()

def onKeyPress(event):
    fig.canvas.mpl_disconnect(cid)
    X = format_input_X(points)
    print(event.key)
    # 'p' key is for perceptron
    if event.key == 'p':
        theta = perceptron(X,y,0.1,1000)
        plot_decision_boundary(theta, 'p')
    # 'v' key is for SVM
    if event.key == 'v':
        theta = SVM(X,y)
        plot_decision_boundary(theta, 'v')
    # 'b' is for BallTree
    if event.key == 'b':
        BallTreeAlg(points)
    # 'e' - for disconnecting the key press event
    if event.key == 'e':
        fig.canvas.mpl_disconnect(pid)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
pid = fig.canvas.mpl_connect('key_press_event', onKeyPress)

plt.show()
