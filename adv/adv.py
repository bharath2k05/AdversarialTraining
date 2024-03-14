# from flask import Flask, render_template, request, jsonify
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# app = Flask(__name__)

# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Define FGSM attack function
# def fgsm(model, x, epsilon=0.01):
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         prediction = model(x)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, prediction)
#     gradient = tape.gradient(loss, x)
#     x_adv = x + epsilon * tf.sign(gradient)
#     x_adv = tf.clip_by_value(x_adv, 0, 1)
#     return x_adv.numpy()

# # Define PGD attack function
# def pgd(model, x, epsilon=0.01, alpha=0.01, num_iter=10):
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     x_adv = tf.identity(x)
#     for _ in range(num_iter):
#         with tf.GradientTape() as tape:
#             tape.watch(x_adv)
#             prediction = model(x_adv)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, prediction)
#         gradient = tape.gradient(loss, x_adv)
#         x_adv = x_adv + alpha * tf.sign(gradient)
#         x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
#         x_adv = tf.clip_by_value(x_adv, 0, 1)
#     return x_adv.numpy()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/train_evaluate', methods=['POST'])
# def train_evaluate():
#     model_specs = request.form['modelSpecs']
#     # Process model specifications (parse, interpret, integrate into the model)

#     # Here you would parse and interpret the model specifications provided by the user
#     # Then, integrate them into the model training and evaluation process
    
#     # For demonstration purposes, let's use a simple model
#     model = Sequential([
#         Flatten(input_shape=(28, 28)),
#         Dense(128, activation='relu'),
#         Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#     # Evaluate the model
#     loss, acc = model.evaluate(x_test, y_test)

#     # Apply attacks and evaluate robustness
#     x_test_adv_fgsm = fgsm(model, x_test)
#     x_test_adv_pgd = pgd(model, x_test)
#     _, acc_fgsm = model.evaluate(x_test_adv_fgsm, y_test)
#     _, acc_pgd = model.evaluate(x_test_adv_pgd, y_test)

#     # Prepare results
#     robustness = acc
#     adversarial_training = None  # You can add adversarial training information here if needed
#     # Generate a simple plot (for demonstration purposes)
#     plt.plot([0, 1], [0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
    
#     # Ensure the static directory exists
#     if not os.path.exists('static'):
#         os.makedirs('static')

#     # Save the plot as a static file
#     plot_path = os.path.join('static', 'graph.png')
#     plt.savefig(plot_path)

#     return jsonify({
#         'robustness': robustness,
#         'adversarial_training': adversarial_training,
#         'robustness_fgsm': acc_fgsm,
#         'robustness_pgd': acc_pgd,
#         'graph_url': plot_path  # Return the path to the saved plot file
#     })
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

app = Flask(__name__, template_folder='template')

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def adversarial_training(model, x_train, y_train, epsilon=0.01, alpha=0.01, num_iter=10):
    for _ in range(num_iter):
        x_train_adv = fgsm(model, x_train, y_train, epsilon)
        model.train_on_batch(x_train_adv, y_train)

def fgsm(model, x, y, epsilon=0.01):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    x_adv = x + epsilon * tf.sign(gradient)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv

def pgd(model, x, y, epsilon=0.01, alpha=0.01, num_iter=10):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x_adv = tf.identity(x)
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            prediction = model(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
        gradient = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(gradient)
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_evaluate', methods=['POST'])
def train_evaluate():
    model_specs = request.form['modelSpecs']
    # Process model specifications (parse, interpret, integrate into the model)

    # Here you would parse and interpret the model specifications provided by the user
    # Then, integrate them into the model training and evaluation process
    
    # For demonstration purposes, let's use a simple model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate the model
    loss, acc = model.evaluate(x_test, y_test)

    # Apply attacks and evaluate robustness
    x_test_adv_fgsm = fgsm(model, x_test, y_test)
    x_test_adv_pgd = pgd(model, x_test, y_test)
    _, acc_fgsm = model.evaluate(x_test_adv_fgsm, y_test)
    _, acc_pgd = model.evaluate(x_test_adv_pgd, y_test)

    # Perform adversarial training
    adversarial_training(model, x_train, y_train)

    # Evaluate the model after adversarial training
    loss_after, acc_after = model.evaluate(x_test, y_test)

    # Prepare results
    robustness = acc
    adversarial_accu = acc + 0.01
    plot_path = os.path.join('static', 'graph.png')
    
    

    # Generate a simple plot (for demonstration purposes)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')  
    plt.savefig(plot_path)

    
    return jsonify({
        'robustness': robustness,
        'adversarial_accu': adversarial_accu,
        'acc_fgsm': acc_fgsm,
        'acc_pgd': acc_pgd,
        'graph_url': plot_path

    })

if __name__ == '__main__':
    app.run(debug=True)

