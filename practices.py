#### Practice 1 #### 
import json

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print("Error: Failed to decode JSON from the file:", e)

    return data

def count_dropped_events(data, NormalizedEvent):
    dropped = len(data['events']) - len(NormalizedEvent['events'])
    return dropped

def parse_json_file(data):
    NormalizedEvent = {"events": []}

    for event in data['events']:
        if event['task_id'] is not None and event['worker_id'] is not None and event['label'] is not None and event['created_at_ms'] is not None:
            label = event["label"].strip()
            if len(label) != 0:
                label = label.lower()
            else:
                continue
            duration_ms = event.get("duration_ms") or 0    
            NormalizedEvent["events"].append({
                    'task_id': event['task_id'],
                    'worker_id': event['worker_id'],
                    'label': label,
                    'created_at_ms': event['created_at_ms'],
                    'duration_ms': duration_ms
                })
    dropped = count_dropped_events(data, NormalizedEvent)
    return NormalizedEvent, dropped

def totalTasks(data):
    totalTask = set()
    for event in data['events']:
        totalTask.add(event['task_id'])
    print(totalTask)
    return len(totalTask)

def compute_statistics(normalized_data):
    """
    Compute statistics for normalized events.
    
    Returns:
    - totalTasks: number of unique taskIds
    - totalEvents: total number of events
    - labelCounts: count by label
    - topWorkersByEvents: top 3 workers by event count (desc, tie-break workerId asc)
    - avgDurationMsByLabel: average duration per label (rounded to nearest int)
    - tasksWithDisagreement: tasks where there are 2+ distinct labels across events
    """
    events = normalized_data['events']
    
    # 1. totalTasks: number of unique taskIds
    unique_task_ids = set(event['task_id'] for event in events)
    totalTasks = len(unique_task_ids)
    
    # 2. totalEvents
    totalEvents = len(events)
    
    # 3. labelCounts: count by label
    labelCounts = {}
    for event in events:
        label = event['label']
        labelCounts[label] = labelCounts.get(label, 0) + 1
    
    # 4. topWorkersByEvents: top 3 workers by event count (desc, tie-break workerId asc)
    worker_counts = {}
    for event in events:
        worker_id = event['worker_id']
        worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
    
    # Sort by count descending, then by worker_id ascending for tie-breaking
    sorted_workers = sorted(
        worker_counts.items(),
        key=lambda x: (-x[1], x[0])  # Negative count for descending, worker_id for ascending
    )
    topWorkersByEvents = [
        {"worker_id": worker_id, "event_count": count}
        for worker_id, count in sorted_workers[:3]
    ]
    
    # 5. avgDurationMsByLabel: average duration per label (rounded to nearest int)
    label_durations = {}
    label_counts_for_avg = {}
    for event in events:
        label = event['label']
        duration_ms = event.get('duration_ms', 0)
        # Convert to int if it's a string
        if isinstance(duration_ms, str):
            duration_ms = int(duration_ms) if duration_ms else 0
        else:
            duration_ms = int(duration_ms) if duration_ms else 0
        
        label_durations[label] = label_durations.get(label, 0) + duration_ms
        label_counts_for_avg[label] = label_counts_for_avg.get(label, 0) + 1
    
    avgDurationMsByLabel = {
        label: round(total_duration / label_counts_for_avg[label])
        for label, total_duration in label_durations.items()
    }
    
    # 6. tasksWithDisagreement: tasks where there are 2+ distinct labels across events
    task_labels = {}
    for event in events:
        task_id = event['task_id']
        label = event['label']
        if task_id not in task_labels:
            task_labels[task_id] = set()
        task_labels[task_id].add(label)
    
    tasksWithDisagreement = [
        task_id for task_id, labels in task_labels.items()
        if len(labels) >= 2
    ]
    # Sort for consistent output
    tasksWithDisagreement.sort()
    
    statistics = {
        "totalTasks": totalTasks,
        "totalEvents": totalEvents,
        "labelCounts": labelCounts,
        "topWorkersByEvents": topWorkersByEvents,
        "avgDurationMsByLabel": avgDurationMsByLabel,
        "tasksWithDisagreement": tasksWithDisagreement
    }
    
    return statistics

data = read_json_file('sample1.json')
normalized_data, dropped_count = parse_json_file(data)

assert dropped_count == 2, f"Expected 2 dropped rows, got {dropped_count}"
print("Normalized Data:")
print(json.dumps(normalized_data, indent=4))

print("\n" + "="*50)
print("Statistics:")
stats = compute_statistics(normalized_data)
print(json.dumps(stats, indent=4))


#### Practice 2 #### 
import json

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print("Error: Failed to decode JSON from the file:", e)

    return data

def normalize_data(data):
    normalized_data = {"predictions": []}
    valid_values = ["pos", "neg"]
    
    for prediction in data['predictions']:
        y_true = prediction.get('y_true')
        y_pred = prediction.get('y_pred')
        
        if (y_true in valid_values and y_pred in valid_values):
            if prediction['id'] is not None and y_true is not None and y_pred is not None and prediction['score'] is not None:
                normalized_data['predictions'].append({
                    'id': prediction['id'],
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'score': prediction['score']
                })
    
    return normalized_data

def compute_metrics(data):
    predictions = data['predictions']
    
    # Initialize confusion matrix (treating "pos" as positive)
    TP = 0  # True Positive: y_true="pos", y_pred="pos"
    FP = 0  # False Positive: y_true="neg", y_pred="pos"
    TN = 0  # True Negative: y_true="neg", y_pred="neg"
    FN = 0  # False Negative: y_true="pos", y_pred="neg"
    
    # Compute confusion matrix
    for pred in predictions:
        y_true = pred['y_true']
        y_pred = pred['y_pred']
        
        if y_true == "pos" and y_pred == "pos":
            TP += 1
        elif y_true == "neg" and y_pred == "pos":
            FP += 1
        elif y_true == "neg" and y_pred == "neg":
            TN += 1
        elif y_true == "pos" and y_pred == "neg":
            FN += 1
    
    # Compute metrics
    total = TP + FP + TN + FN
    
    # Accuracy
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Round to 3 decimals
    accuracy = round(accuracy, 3)
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)
    
    metrics = {
        "confusion_matrix": {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Compute Precision@K if score is available on ≥1 row
    scores_available = any(pred.get('score') is not None for pred in predictions)
    if scores_available:
        # Sort by score descending (convert to float if string)
        sorted_predictions = sorted(
            predictions,
            key=lambda x: float(x['score']) if isinstance(x.get('score'), str) else (x.get('score') or 0),
            reverse=True
        )

        n = len(sorted_predictions)
        K_max = min(5, n)
        
        precision_at_k = {}
        for K in range(1, K_max + 1):
            # Precision@K: of the top K items, how many are actually positive (y_true="pos")
            # This is a ranking metric: proportion of relevant items in top K
            relevant_at_k = sum(1 for pred in sorted_predictions[:K] if pred['y_true'] == "pos")
            prec_k = relevant_at_k / K if K > 0 else 0.0
            precision_at_k[f"P@{K}"] = round(prec_k, 3)
        
        metrics["precision_at_k"] = precision_at_k
    
    return metrics

json_data = read_json_file('sample2.json')
result = normalize_data(json_data)
print("Normalized Data:")
print(json.dumps(result, indent=4))

print("\n" + "="*50)
print("Metrics:")
metrics = compute_metrics(result)
print(json.dumps(metrics, indent=4))




#### Practice 3 #### 
import json

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print("Error: Failed to decode JSON from the file:", e)

    return data

def rules(data):
    logs = data['logs']
    new_logs = {"logs": []}
    for log in logs:
        # Check if latency_ms is a valid number
        latency_ms = log.get('latency_ms')
        
        if isinstance(latency_ms, str):
            if latency_ms.isdigit():  # Check if string contains only digits
                latency = int(latency_ms)
            else:
                continue  # Skip if not a valid number string (e.g., "bad")
        elif isinstance(latency_ms, int):
            latency = latency_ms
   
        
        if log['status'] is not None and latency_ms is not None and log['service'] is not None and log['endpoint'] is not None:
            # Remove query parameters from endpoint (everything after "?")
            endpoint = log['endpoint'].split('?')[0]

            new_logs['logs'].append({
                'service': log['service'],
                'endpoint': endpoint, 
                'status': log['status'],
                'latency_ms': latency
            })
    return new_logs

def compute_statistics(norm_data):
    import math
    from collections import defaultdict
    
    logs = norm_data['logs']
    
    # Group logs by service
    service_logs = defaultdict(list)
    for log in logs:
        service_logs[log['service']].append(log)
    
    # 1. Error rate per service: 5xx / total
    error_rate_per_service = {}
    for service, service_log_list in service_logs.items():
        total = len(service_log_list)
        error_5xx = sum(1 for log in service_log_list if log['status'] >= 500)
        error_rate = error_5xx / total if total > 0 else 0.0
        error_rate_per_service[service] = round(error_rate, 3)
    
    # 2. P95 latency per service (sort latencies and pick ceil(0.95*n)-1)
    p95_latency_per_service = {}
    for service, service_log_list in service_logs.items():
        latencies = [log['latency_ms'] for log in service_log_list]
        latencies.sort()
        n = len(latencies)
        if n > 0:
            index = math.ceil(0.95 * n) - 1
            index = max(0, min(index, n - 1))  # Ensure index is within bounds
            p95_latency_per_service[service] = latencies[index]
        else:
            p95_latency_per_service[service] = 0
    
    # 3. Top 3 slowest endpoints overall by avg latency (min 2 samples)
    endpoint_latencies = defaultdict(list)
    for log in logs:
        endpoint = log['endpoint']
        endpoint_latencies[endpoint].append(log['latency_ms'])
    
    # Calculate average latency per endpoint (only for endpoints with >= 2 samples)
    endpoint_avg_latency = {}
    for endpoint, latencies in endpoint_latencies.items():
        if len(latencies) >= 2:  # Minimum 2 samples
            avg_latency = sum(latencies) / len(latencies)
            endpoint_avg_latency[endpoint] = avg_latency
    
    # Sort by average latency descending and take top 3
    sorted_endpoints = sorted(
        endpoint_avg_latency.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    top_3_slowest_endpoints = [
        {"endpoint": endpoint, "avg_latency_ms": round(avg_latency, 2)}
        for endpoint, avg_latency in sorted_endpoints
    ]
    
    return {
        "error_rate_per_service": error_rate_per_service,
        "p95_latency_per_service": p95_latency_per_service,
        "top_3_slowest_endpoints": top_3_slowest_endpoints
    }

json_data = read_json_file('sample3.json')
normalized_data = rules(json_data)
print("Normalized Data:")
print(json.dumps(normalized_data, indent=4))

print("\n" + "="*50)
print("Statistics:")
stats = compute_statistics(normalized_data)
print(json.dumps(stats, indent=4))

