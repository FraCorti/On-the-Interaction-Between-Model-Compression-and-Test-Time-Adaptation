import numpy as np


def format_map_results(data):
    formatted_data = {}

    for corruption_type, row_list in data.items():
        formatted_rows = []
        for row in row_list:
            if isinstance(row, list) and len(row) > 0:
                inner_dict = row[0]

                clean_values = [round(float(val), 4) for val in inner_dict.values()]
                formatted_rows.append(clean_values)

        formatted_data[corruption_type] = formatted_rows

    return formatted_data


def format_numpy_floats_map(results):
    rounded_data = {}
    for key, value_list in results.items():
        # Convert numpy float to python float and round
        # Robustly handle both scalars and lists (vectors) in the value_list
        try:
             rounded_list = []
             for x in value_list:
                 if isinstance(x, list):
                     rounded_list.append([round(float(v), 4) for v in x])
                 elif isinstance(x, (np.ndarray, tuple)): # Handle other iterables just in case
                     rounded_list.append([round(float(v), 4) for v in x])
                 else:
                     rounded_list.append(round(float(x), 4))
             rounded_data[key] = rounded_list
        except Exception as e:
            # Fallback to avoid crashing the whole run
             print(f"Warning: format_numpy_floats_map failed for key {key}: {e}")
             rounded_data[key] = value_list

    return rounded_data


def format_metric_map(metric_data_per_ratio, ratios, corruptions):
    result_map = {c: [] for c in corruptions}
    sorted_ratios = sorted(ratios)

    for c_idx, c_name in enumerate(corruptions):

        ratio_values = []
        for r in sorted_ratios:
            if r not in metric_data_per_ratio:
                ratio_values.append([])  # Missing ratio?
                continue

            # Access data for this corruption
            # Check if it's a list or dict
            data_at_ratio = metric_data_per_ratio[r]

            vals = []
            if isinstance(data_at_ratio, list):
                if c_idx < len(data_at_ratio):
                    vals = data_at_ratio[c_idx]
            elif isinstance(data_at_ratio, dict):
                if c_idx in data_at_ratio:
                    vals = data_at_ratio[c_idx]

            processed_val = vals
            if isinstance(vals, list) and len(vals) > 0:
                # Check if elements are scalars or vectors/lists
                if isinstance(vals[0], (int, float, np.number)):
                    processed_val = np.mean(vals)
                elif isinstance(vals[0], (list, np.ndarray)):
                    # List of vectors. Aggregate.
                    # We need L (length). Assume consistent.
                    L_vec = len(vals[0])
                    # Use the helper if available or simple mean
                    try:
                        processed_val = np.nanmean(np.vstack(vals), axis=0).tolist()
                    except:
                        processed_val = vals  # Fallback

            ratio_values.append(processed_val)

            result_map[c_name] = ratio_values
    return result_map