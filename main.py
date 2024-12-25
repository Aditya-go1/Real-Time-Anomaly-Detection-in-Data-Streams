from collections import deque
from river import anomaly, compose, preprocessing, metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- Dynamic Threshold Calculation ----
def calculate_hybrid_threshold(scores, static_threshold, alpha=0.1, percentile=97):
    """
    Combines static threshold with dynamic adjustment using Exponential Moving Average (EMA).
    :param scores: Recent anomaly scores in a rolling window.
    :param static_threshold: Pre-calculated static threshold from warmup data.
    :param alpha: Weight factor for dynamic threshold in the hybrid calculation.
    :param percentile: Percentile for dynamic threshold calculation.
    :return: Updated hybrid threshold.
    """
    dynamic_threshold = np.percentile(scores, percentile)
    return alpha * dynamic_threshold + (1 - alpha) * static_threshold

# ---- Data Stream Generation ----
def financial_data_with_drift_and_seasonality(n_samples=20000, anomaly_rate=0.01):
    """
    Simulates a data stream with seasonal trends, drift, and noise, and injects anomalies.
    :param n_samples: Number of data points in the stream.
    :param anomaly_rate: Proportion of anomalies to introduce.
    :return: Simulated data stream and indices of true anomalies.
    """
    np.random.seed(42)
    timestamps = np.arange(n_samples)

    # Simulate concept drift (gradual trend) and random noise
    growth = 2 * np.sin(2 * np.pi * timestamps / 5000)
    noise = np.random.normal(0.1, 0.3, n_samples)

    # Base signal with seasonal pattern
    seasonality = np.sin(2 * np.pi * timestamps / 500)

    # Combine seasonality, drift, and noise
    data_stream = seasonality.copy() + growth + noise

    # Inject anomalies by adding large random deviations
    anomaly_indices = np.random.choice(n_samples, size=int(anomaly_rate * n_samples), replace=False)
    data_stream[anomaly_indices] += np.random.normal(4, 0.5, len(anomaly_indices))

    return data_stream, anomaly_indices

# ---- Anomaly Detection with Hybrid Thresholding ----
def anomaly_detection_with_hybrid_threshold(data_stream, window_size=50, n_trees=100, update_freq=5000, warmup_points=1000, alpha=0.1):
    """
    Detects anomalies in a data stream using Half-Space Trees and a hybrid thresholding strategy.
    :param data_stream: Input data stream for detection.
    :param window_size: Window size for Half-Space Trees.
    :param n_trees: Number of trees in the ensemble.
    :param update_freq: Frequency of threshold recalculation.
    :param warmup_points: Number of initial points used to calculate the static threshold.
    :param alpha: Weight for blending static and dynamic thresholds.
    :return: Predictions (0/1) and hybrid thresholds over time.
    """
    # Initialize the Half-Space Trees model with MinMaxScaler preprocessing
    model = compose.Pipeline(
        preprocessing.MinMaxScaler(),
        anomaly.HalfSpaceTrees(n_trees=n_trees, height=8, window_size=window_size, seed=42)
    )

    # Metrics for evaluating the model
    auc = metrics.ROCAUC()
    f1 = metrics.F1()

    # Data structures for storing results
    anomaly_scores = deque(maxlen=warmup_points)  # Rolling window for anomaly scores
    predictions = []  # List to store anomaly predictions (0 or 1)
    thresholds = []  # List to store threshold values over time
    threshold = None
    static_threshold = None  # Initial static threshold

    # Iterate through the data stream
    for i, x in enumerate(data_stream):
        features = {'value': x}  # Wrap the data point into a feature dictionary

        # Update the model with the current data point
        model.learn_one(features)

        # Compute the anomaly score for the current data point
        score = model.score_one(features)
        anomaly_scores.append(score)

        # Static threshold calculation after warmup period
        if i == warmup_points - 1:
            static_threshold = np.percentile(anomaly_scores, 97)  # Set static threshold at 97th percentile
            threshold = static_threshold
            thresholds.append(static_threshold)
            print("Initial Static Threshold:", threshold)

        # Periodic recalculation of hybrid threshold
        if i >= update_freq and i % update_freq == warmup_points:
            threshold = calculate_hybrid_threshold(anomaly_scores, static_threshold, alpha, 97)
            thresholds.append(threshold)
            print("Updated Hybrid Threshold:", threshold)

        # Anomaly detection based on threshold
        is_anomaly = 1 if threshold and score > threshold else 0
        predictions.append(is_anomaly)

            # Update AUC and F1 metrics
        label = 1 if i in anomaly_indices else 0
        auc.update(label, score)
        f1.update(label, is_anomaly)
    
    # Final Metrics
    print(f"ROCAUC: {auc}")
    print(f"F1 Score: {f1}")
    #ROCAUC: ROCAUC: 81.64%
    #F1 Score: F1: 96.64%

    return predictions, thresholds

# Visualization Function: Live Plot with Dynamic Threshold
def live_plot_dynamic_threshold(data_stream, predictions, window_size=500):
    """
    Creates a live plot to visualize the data stream and dynamically highlights detected anomalies in real-time.

    Parameters:
    - data_stream: The input data stream (array-like) to be visualized.
    - predictions: A list of anomaly predictions where 1 represents an anomaly.
    - window_size: Number of data points to display in the plot at any given time (default is 500).
    """

    # Set up the plot figure and axis
    fig, ax = plt.subplots(figsize=(18, 12))

    # Function to update the plot for each frame of the animation
    def update(frame):
        # Clear the previous frame's data
        ax.clear()

        # Plot the data stream up to the current frame
        ax.plot(range(frame + 1), data_stream[:frame + 1], color='blue', label="Data Stream")

        # Identify and highlight detected anomalies in red
        anomalies = [i for i in range(frame + 1) if predictions[i] == 1]
        ax.scatter(anomalies, data_stream[anomalies], color='red', label="Detected Anomalies", marker='x')

        # Set plot title and labels
        ax.set_title("Real-Time Anomaly Detection with Dynamic Threshold")
        ax.set_xlabel("Time")  # X-axis represents the time or index of the data point
        ax.set_ylabel("Transaction Value")  # Y-axis represents the transaction value
        ax.legend()  # Add a legend to indicate different plot elements

    # Create an animation object to visualize the live plot
    ani = animation.FuncAnimation(fig, update, frames=len(data_stream), interval=30, repeat=False)

    # Display the animation
    plt.show()


# ---- Generate Synthetic Data ----
data_stream, anomaly_indices = financial_data_with_drift_and_seasonality()

# ---- Run Anomaly Detection ----
predictions, thresholds = anomaly_detection_with_hybrid_threshold(data_stream, window_size=50, n_trees=100)

# ---- Visualize results with live plot ----
#live_plot_dynamic_threshold(data_stream, predictions)

# ---- Static Visualization ----
plt.figure(figsize=(12, 6))
plt.plot(data_stream, label="Data Stream", alpha=0.7)

# Highlight true anomalies (green markers)
#for verifying if the anomalies detected are true or not
plt.scatter(anomaly_indices, data_stream[anomaly_indices], color='green', label="True Anomalies", marker='o')


# Highlight detected anomalies (red markers)
detected_anomalies = [i for i, pred in enumerate(predictions) if pred == 1]
plt.scatter(detected_anomalies, data_stream[detected_anomalies], color='red', label="Detected Anomalies", marker='x')

#plt.plot(anomaly_scores, label="Anomaly Scores", linestyle="--", color="orange", alpha=0.7)

# Add labels and legend
plt.legend()
plt.title("Half-Space Trees Anomaly Detection with Hybrid Thresholding")
plt.xlabel("Time")
plt.ylabel("System Metrics")
plt.show()
