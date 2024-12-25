

---

# Real Time Anomaly Detection in Data Streams Using Half-Space Trees

This repository demonstrates the implementation of an anomaly detection system for a synthetic data stream representing system metrics, utilizing a hybrid thresholding strategy. The model leverages **Half-Space Trees** from the `river` library, a state-of-the-art algorithm for online anomaly detection, and combines it with a **dynamic thresholding** technique that adapts over time. I have also looked into the iForestASD but found this better for real-time anomaly detection.

## Key Features

- **Hybrid Thresholding:** Combines static thresholding (based on a warm-up period) and dynamic adjustments using Exponential Moving Average (EMA) and percentiles.
- **Anomaly Detection:** Uses **Half-Space Trees** for detecting anomalies in data streams with drift and seasonality.
- **Synthetic Data Stream:** Generates a synthetic data stream that simulates financial metrics, including concept drift, seasonality, noise, and injected anomalies.
- **Real-Time Visualization:** A live plot updates in real-time to visualize detected anomalies and dynamic thresholds.
- **Evaluation Metrics:** Provides performance evaluation using **AUC-ROC** and **F1-Score**.

## Requirements

To run the project, you'll need the following Python libraries:

- `river`: For streaming machine learning models, including anomaly detection algorithms.
- `numpy`: For numerical operations and data manipulation.
- `matplotlib`: For data visualization.

You can install these dependencies via `pip`:

```bash
pip install river numpy matplotlib 
```

## Project Structure

The project includes the following components:

- **`calculate_hybrid_threshold` function**: Calculates the hybrid threshold combining static and dynamic thresholds.
- **`financial_data_with_drift_and_seasonality` function**: Simulates a synthetic financial data stream with drift, seasonality, and noise, and injects anomalies.
- **`anomaly_detection_with_hybrid_threshold` function**: Runs the anomaly detection algorithm using Half-Space Trees with dynamic thresholding.
- **`live_plot_dynamic_threshold` function**: Visualizes the results of anomaly detection in real-time, with dynamic thresholds.
- **`main.py`**: Contains the code to generate synthetic data, detect anomalies, and visualize the results.

## Usage

### 1. **Generate Synthetic Data**

The synthetic data stream is generated using the `financial_data_with_drift_and_seasonality` function. It simulates financial transactions, incorporating seasonality, drift, and random noise.

```python
data_stream, anomaly_indices = financial_data_with_drift_and_seasonality()
```

- `data_stream`: The synthetic data points.
- `anomaly_indices`: The indices of the injected anomalies for evaluation.

### 2. **Anomaly Detection**

The `anomaly_detection_with_hybrid_threshold` function performs anomaly detection using the Half-Space Trees algorithm and computes dynamic thresholds.

```python
predictions, thresholds = anomaly_detection_with_hybrid_threshold(data_stream, window_size=50, n_trees=100)
```

Parameters:

- `data_stream`: The synthetic data stream.
- `window_size`: The window size for Half-Space Trees (default is 50).
- `n_trees`: The number of trees in the ensemble (default is 30).
- `update_freq`: Frequency of threshold recalculation (default is 5000).
- `warmup_points`: Number of initial points used to calculate the static threshold (default is 1000).
- `alpha`: The blending weight for the static and dynamic thresholds (default is 0.1).

### 3. **Visualization**

To visualize the results with a static plot, the anomalies are plotted against the data stream. True anomalies are highlighted in green, while detected anomalies are marked in red.

![Static visualiztion with 100 trees](/100trees.png)

Alternatively, you can visualize the detection results in **real-time** with the `live_plot_dynamic_threshold` function:

```python
live_plot_dynamic_threshold(data_stream, predictions)
```

## Metrics

- **ROCAUC:** Measures the model's ability to distinguish between normal and anomalous points.
- **F1 Score:** Harmonic mean of precision and recall, which is particularly useful when dealing with imbalanced datasets.

## Results

Upon testing the anomaly detection on synthetic data, the model achieves the following performance:

- **ROCAUC**: 79.50%
- **F1 Score**: 94.90%

These results demonstrate the effectiveness of combining static and dynamic thresholds in detecting anomalies over time. I tired with other combinations as well.

## Research References

1. **Concept Drift Best Practices** – Neptune AI Blog  
    [Read the article here](https://neptune.ai/blog/concept-drift-best-practices)
    
2. **A Survey on Data Stream Mining and the Applications to Real-Time Anomaly Detection** – ScienceDirect  
    [Read the article here](https://www.sciencedirect.com/science/article/pii/S1474667016314999)
    
3. **Anomaly Detection for Data Streams Based on Isolation Forest Using Scikit-Multiflow** – ResearchGate  
    [Read the article here](https://www.researchgate.net/publication/345786638_Anomaly_Detection_for_Data_Streams_Based_on_Isolation_Forest_Using_Scikit-Multiflow)
    
4. **A Robust Anomaly Detection Algorithm for Data Streams** – IJCAI Proceedings  
    [Read the paper here](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)

These papers provide insights into the mechanisms of anomaly detection, the effectiveness of Half-Space Trees and Isolation Forests , and the strategies for handling concept drift in streaming data.

## Comparison of Half-Space Trees and iForestASD

### **Half-Space Trees (HST)**

- **Real-Time Anomaly Detection**: HST processes data **on a point-by-point basis**, making it ideal for **real-time anomaly detection** in streaming data.
- **Dynamic Thresholding**: Combines static thresholds with dynamic adjustments, adapting well to changing data distributions (concept drift).
- **Scalable and Adaptable**: Effective for high-dimensional data and environments with drift.
- **Strengths**: Suitable for **streaming data with concept drift**, providing continuous updates as new data arrives.

### **iForestASD (Isolation Forest for Data Streams)**

- **Batch Processing**: iForestASD uses a **sliding window** approach and processes data in **batches**, updating periodically rather than on a point-by-point basis.
- **Efficient for Large Datasets**: Highly scalable and computationally efficient, especially for large-scale data streams with relatively stable distributions.
- **Limitations**: Less adaptable to **concept drift** and not as suited for real-time applications where anomalies need to be detected immediately.

### **Summary**

|Feature|**Half-Space Trees (HST)**|**iForestASD**|
|---|---|---|
|**Processing**|Real-time, point-by-point|Batch processing, sliding window|
|**Adaptability to Concept Drift**|High (dynamic thresholding)|Limited (sliding window)|
|**Best Use Case**|Real-time, drift-sensitive data streams|Large-scale, stable data streams|
|**Strengths**|Fast, adaptive to changes, ideal for online detection|Scalable, efficient for large data sets|
|**Weaknesses**|More memory-intensive, higher complexity|Limited real-time performance and drift adaptation|

### **Conclusion**

- **HST** is ideal for **real-time anomaly detection** in data streams that change over time, as it processes data point-by-point.
- **iForestASD** is best suited for large-scale data streams where **batch processing** is acceptable, and concept drift is less of a concern.

## Contact
For questions, suggestions, or contributions, please contact:

- **Aditya Bhadouria**
- [Email](mailto:aditya.bhadouria.official@gmail.com)
- **GitHub**: [github-profile](https://github.com/Aditya-go1)

---
# Real-Time-Anomaly-Detection-in-Data-Streams
