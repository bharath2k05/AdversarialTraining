import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Define attack functions
def fgsm(model, x, y, epsilon=0.01):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    x_adv = x + epsilon * tf.sign(gradient)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()

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
    return x_adv.numpy()

def momentum_iterative_fgsm(model, x, y, epsilon=0.01, alpha=0.01, num_iter=10, decay_factor=1.0):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    decayed_epsilon = epsilon * decay_factor
    x_adv = tf.identity(x)
    g = 0
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            prediction = model(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
        gradient = tape.gradient(loss, x_adv)
        g = decay_factor * g + tf.sign(gradient)
        x_adv = x_adv + alpha * tf.sign(g)
        x_adv = tf.clip_by_value(x_adv, x - decayed_epsilon, x + decayed_epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()

def random_noise(model, x, y, epsilon=0.01):
    x_noise = np.random.uniform(low=-epsilon, high=epsilon, size=x.shape)
    x_adv = x + x_noise
    x_adv = np.clip(x_adv, 0, 1)
    return x_adv

# Attack methods
attack_methods = [
    fgsm,
    pgd,
    momentum_iterative_fgsm,
    random_noise
]

# Evaluate robustness before adversarial training
robustness_before = np.zeros(len(attack_methods))
for j, attack_method in enumerate(attack_methods):
    x_test_adv = attack_method(model, x_test, y_test)
    _, acc = model.evaluate(x_test_adv, y_test, verbose=0)
    robustness_before[j] = acc

# Adversarial training
for _ in range(3):  # Perform adversarial training for 3 epochs
    for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(32):
        with tf.GradientTape() as tape:
            predictions = model(x_batch)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Evaluate robustness after adversarial training
robustness_after = np.zeros(len(attack_methods))
for j, attack_method in enumerate(attack_methods):
    x_test_adv = attack_method(model, x_test, y_test)
    _, acc = model.evaluate(x_test_adv, y_test, verbose=0)
    robustness_after[j] = acc

# Create Plotly figure
fig = make_subplots(rows=1, cols=1)

# Add traces
fig.add_trace(
    go.Scatter(x=np.arange(len(attack_methods)), y=robustness_before, mode='lines+markers', name='Before Adversarial Training', marker=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(len(attack_methods)), y=robustness_after, mode='lines+markers', name='After Adversarial Training', marker=dict(color='orange')),
    row=1, col=1
)

# Update layout
fig.update_layout(
    title='Comparison of Adversarial Robustness before and after Adversarial Training',
    xaxis_title='Attack Methods',
    yaxis_title='Robustness',
    xaxis_tickvals=np.arange(len(attack_methods)),
    xaxis_ticktext=['FGSM', 'PGD', 'Momentum Iterative FGSM', 'Random Noise'],
    legend=dict(x=0, y=1, traceorder='normal')
)

# Show the plot
fig.show()
fig.write_html("advrr.html")