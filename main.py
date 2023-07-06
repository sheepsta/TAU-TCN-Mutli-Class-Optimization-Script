import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime

max_accuracy = 0
max_sublen = 0
max_snip = 0
matrix_minlen = 10000
matrix_maxlen = 0
subsequence_length = 50
time_start = datetime.datetime.now()

num_hex_files = 127
num_cyl_files = 200
num_david_files = 86

accuracy_data = []

cylinder_data = []
hexagon_data = []
david_data = []

for i in range(1, num_cyl_files+1):
    cylinder_data.append(np.load(f'cylinder{i}.npy'))

for i in range(1, num_hex_files+1):
    hexagon_data.append(np.load(f'hexagon{i}.npy'))

for i in range(1, num_david_files+1):
    david_data.append(np.load(f'david{i}.npy'))

for matrix in hexagon_data:
    if matrix.shape[0] < matrix_minlen:
        matrix_minlen = matrix.shape[0]
    if matrix.shape[0] > matrix_maxlen:
        matrix_maxlen = matrix.shape[0]

for matrix in cylinder_data:
    if matrix.shape[0] < matrix_minlen:
        matrix_minlen = matrix.shape[0]
    if matrix.shape[0] > matrix_maxlen:
        matrix_maxlen = matrix.shape[0]

for matrix in david_data:
    if matrix.shape[0] < matrix_minlen:
        matrix_minlen = matrix.shape[0]
    if matrix.shape[0] > matrix_maxlen:
        matrix_maxlen = matrix.shape[0]

total_steps = (matrix_minlen-50)*30
steps_complete = 0
print(f"The total steps required for optimization is {total_steps}")

for y in range(50, matrix_minlen):
    for z in range(20, 50):
        subsequence_length = z

        # Load the data
        cylinder_data = []
        hexagon_data = []
        david_data = []

        for i in range(1, num_cyl_files+1):
            try:
                cylinder_data.append(np.load(f'cylinder{i}.npy')[0:y, :])
            except:
                pass

        for i in range(1, num_hex_files+1):
            try:
                hexagon_data.append(np.load(f'hexagon{i}.npy')[0:y, :])
            except:
                pass

        for i in range(1, num_david_files+1):
            try:
                david_data.append(np.load(f'david{i}.npy')[0:y, :])
            except:
                pass

        # Preprocess the data
        def preprocess_data(data):
            subsequences = []
            scaler = StandardScaler()

            for matrix in data:
                length = matrix.shape[0]
                num_subsequences = length // subsequence_length
                reshaped_matrix = matrix[:num_subsequences*subsequence_length, :].reshape(num_subsequences, subsequence_length, 7)

                # Remove null values and filter subsequence
                valid_matrices = []
                for subseq in reshaped_matrix:
                    if not np.isnan(subseq).any() and subseq.shape[0] == subsequence_length:
                        filtered_subseq = apply_low_pass_filter(subseq)
                        valid_matrices.append(filtered_subseq)

                if valid_matrices:
                    normalized_matrices = scaler.fit_transform(np.array(valid_matrices).reshape(-1, 7))
                    subsequences.append(normalized_matrices.reshape(len(valid_matrices), subsequence_length, 7))

            return np.concatenate(subsequences)

        def apply_low_pass_filter(matrix):
            # Apply a low-pass filter to the matrix
            b, a = butter(4, 0.1, btype='lowpass', analog=False)
            filtered_matrix = filtfilt(b, a, matrix, axis=0)
            return filtered_matrix

        cylinder_data = preprocess_data(cylinder_data)
        hexagon_data = preprocess_data(hexagon_data)
        david_data = preprocess_data(david_data)

        # Create labels
        cylinder_labels = np.zeros(cylinder_data.shape[0])
        hexagon_labels = np.ones(hexagon_data.shape[0])
        david_labels = np.full(david_data.shape[0], 2)

        # Merge data and labels
        data = np.concatenate((cylinder_data, hexagon_data, david_data), axis=0)
        labels = np.concatenate((cylinder_labels, hexagon_labels, david_labels), axis=0)

        # Split the data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

        # Build the TCN CNN model
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(subsequence_length, 7)),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')  # Updated for the three classes (cylinder, hexagon, and Star of David)
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Define early stopping criteria
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # Train the model with early stopping
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        accuracy_data.append([subsequence_length, y, accuracy])

        steps_complete += 1
        process_progress = steps_complete/total_steps
        time_difference = datetime.datetime.now() - time_start
        time_difference = time_difference.total_seconds()
        time_per_step_live = (time_difference)/steps_complete
        total_time_eta = time_per_step_live * total_steps - time_difference
        print(f"-----The process is {round(process_progress*100, 2)}% complete. ETA: {total_time_eta/60} min.----")

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_sublen = subsequence_length
            max_snip = y

    print(f"Maximum accuracy thus far achieved with buffer length {max_sublen} and snip of {max_snip}, accuracy: {max_accuracy}")

ax = plt.axes()
for x in accuracy_data:
    ax.arrow(x[0], x[1], math.cos(x[2]*math.pi/2)*.66, math.sin(x[2]*math.pi/2)*.66, head_width=0.1, head_length=.3)
plt.ylim(50,matrix_minlen)
plt.xlim(20,50)
plt.xlabel('Subsequence Length')
plt.ylabel('Matrix Cutoff')
plt.title(f'TCN Accuracy Vector Field for Subsequence Len {subsequence_length} and Matrix Cutoff {y}')
plt.show()
