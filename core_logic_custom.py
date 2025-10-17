import os
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import gaussian_filter, sobel, laplace, generic_filter, binary_opening, binary_closing, label as scipy_label
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter, median_filter

AUTO_HAZARD_THRESHOLD = -8000
WINDOW_SIZE = 15

def load_dem(path, target_shape=(600, 1200)):
    """Load and resample Digital Elevation Model."""
    try:
        with rasterio.open(path) as src:
            print(f"Original DEM shape: {src.shape}, CRS: {src.crs}")
            elevation = src.read(1, out_shape=target_shape, resampling=Resampling.average)
        
        elevation = elevation.astype(np.float32)
        
        if hasattr(elevation, 'mask'):
            elevation = np.ma.filled(elevation, np.nan)
        
        if np.all(np.isnan(elevation)):
            raise ValueError("DEM contains only NaN values")
            
        return elevation
        
    except Exception as e:
        raise RuntimeError(f"Failed to load DEM: {e}")

def compute_terrain_roughness(elevation, window_size=WINDOW_SIZE):
    """Compute terrain roughness (std dev in local neighborhood)."""
    def local_std(values):
        return np.std(values)
    
    roughness = generic_filter(elevation, local_std, size=window_size, mode='reflect')
    return roughness

def compute_local_slope_stats(slope, window_size=WINDOW_SIZE):
    """Compute local slope statistics."""
    mean_slope = uniform_filter(slope, size=window_size, mode='reflect')
    max_slope = maximum_filter(slope, size=window_size, mode='reflect')
    return mean_slope, max_slope

def compute_elevation_range(elevation, window_size=WINDOW_SIZE):
    """Compute local elevation range."""
    max_elev = maximum_filter(elevation, size=window_size, mode='reflect')
    min_elev = minimum_filter(elevation, size=window_size, mode='reflect')
    elev_range = max_elev - min_elev
    return elev_range

def compute_features(elevation):
    """Compute comprehensive terrain features."""
    valid_mask = ~np.isnan(elevation)
    if not np.any(valid_mask):
        raise ValueError("No valid elevation data found")
    
    elev_filled = np.where(valid_mask, elevation, np.nanmean(elevation))
    
    print("  Computing basic features (slope, curvature)...")
    smoothed = gaussian_filter(elev_filled, sigma=3)
    
    slope_x = sobel(smoothed, axis=1)
    slope_y = sobel(smoothed, axis=0)
    slope = np.hypot(slope_x, slope_y)
    
    curvature = laplace(smoothed)
    
    print("  Computing spatial features (roughness, local statistics)...")
    roughness = compute_terrain_roughness(smoothed, window_size=WINDOW_SIZE)
    mean_slope, max_slope = compute_local_slope_stats(slope, window_size=WINDOW_SIZE)
    elev_range = compute_elevation_range(smoothed, window_size=WINDOW_SIZE)
    
    features = {
        'elevation': np.where(valid_mask, elevation, np.nan),
        'slope': np.where(valid_mask, slope, np.nan),
        'curvature': np.where(valid_mask, curvature, np.nan),
        'roughness': np.where(valid_mask, roughness, np.nan),
        'mean_slope': np.where(valid_mask, mean_slope, np.nan),
        'max_slope': np.where(valid_mask, max_slope, np.nan),
        'elev_range': np.where(valid_mask, elev_range, np.nan)
    }
    
    print(f"  ✓ Computed {len(features)} feature layers")
    return features

def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print(f"WARNING: Expected list in {label_path}, got {type(data)}")
            return []
            
        valid_labels = []
        for item in data:
            if isinstance(item, list) and len(item) == 3:
                x, y, label = item
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and label in [0, 1]:
                    valid_labels.append([int(x), int(y), int(label)])
                else:
                    print(f"WARNING: Invalid label format: {item}")
            else:
                print(f"WARNING: Invalid label structure: {item}")
                
        return valid_labels
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {label_path}: {e}")
        return []
    except Exception as e:
        print(f"ERROR: Failed to load labels: {e}")
        return []

def save_labels(label_path, labels):
    try:
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"Saved {len(labels)} labeled points to {label_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to save labels: {e}")

def build_training_data(features_dict, labels):
    """Build feature matrix and target vector from labeled points."""
    if not labels:
        raise ValueError("No labels provided for training")
    
    X, Y = [], []
    h, w = features_dict['elevation'].shape
    
    feature_names = ['elevation', 'slope', 'curvature', 'roughness', 
                     'mean_slope', 'max_slope', 'elev_range']
    
    for x, y, label in labels:
        if not (0 <= x < w and 0 <= y < h):
            print(f"WARNING: Label coordinates ({x}, {y}) out of bounds, skipping")
            continue
            
        feature_vector = []
        valid = True
        
        for feat_name in feature_names:
            feat_value = features_dict[feat_name][y, x]
            if np.isnan(feat_value):
                valid = False
                break
            feature_vector.append(feat_value)
        
        if not valid:
            print(f"WARNING: NaN values at ({x}, {y}), skipping")
            continue
            
        X.append(feature_vector)
        Y.append(label)
    
    if not X:
        raise ValueError("No valid training samples after filtering")
    
    return np.array(X), np.array(Y), feature_names

def train_model(X, Y, existing_model=None):
    if len(X) < 2:
        raise ValueError("Need at least 2 training samples")
    
    if existing_model and 'data' in existing_model:
        try:
            X_prev, Y_prev = existing_model['data']
            X = np.vstack([X_prev, X])
            Y = np.concatenate([Y_prev, Y])
            print(f"Combined with {len(X_prev)} previous samples")
        except Exception as e:
            print(f"WARNING: Failed to load previous data: {e}")
    
    unique, counts = np.unique(Y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X, Y)
    
    train_pred = clf.predict(X)
    train_acc = accuracy_score(Y, train_pred)
    print(f"Training accuracy: {train_acc:.3f}")
    
    return clf, (X, Y)

def smooth_probability_map(prob_map, sigma=5):
    smoothed = gaussian_filter(prob_map, sigma=sigma)
    smoothed = np.clip(smoothed, 0, 1)
    return smoothed

def post_process_predictions(prob_map, threshold=0.9, min_area=100):
    safe_mask = (prob_map >= threshold).astype(np.uint8)
    structuring_element = np.ones((7, 7), dtype=np.uint8)
    safe_mask = binary_opening(safe_mask, structure=structuring_element, iterations=3)
    safe_mask = binary_closing(safe_mask, structure=structuring_element, iterations=2)
    labeled_array, num_features = scipy_label(safe_mask)
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_array == region_id)
        region_size = np.sum(region_mask)
        
        if region_size < min_area:
            safe_mask[region_mask] = 0
    
    return safe_mask.astype(np.uint8)

def predict_safe_areas(clf, features_dict, threshold=0.9, apply_postprocess=True, min_area=100):
    h, w = features_dict['elevation'].shape
    
    feature_names = ['elevation', 'slope', 'curvature', 'roughness', 
                     'mean_slope', 'max_slope', 'elev_range']
    feature_list = [features_dict[name].ravel() for name in feature_names]
    features = np.stack(feature_list, axis=1)
    
    valid_mask = ~np.any(np.isnan(features), axis=1)
    prob = np.zeros(h * w)
    
    if np.any(valid_mask):
        valid_features = features[valid_mask]
        valid_prob = clf.predict_proba(valid_features)[:, 1]
        prob[valid_mask] = valid_prob
    
    prob_map = prob.reshape(h, w)
    
    print("  Applying spatial smoothing for contiguous zones...")
    prob_map_smoothed = smooth_probability_map(prob_map, sigma=8)

    if apply_postprocess:
        print(f"  Applying spatial post-processing (min area: {min_area} pixels)...")
        mask = post_process_predictions(prob_map_smoothed, threshold=threshold, min_area=min_area)
        
        original_safe = np.sum(prob_map_smoothed >= threshold)
        processed_safe = np.sum(mask)
        if original_safe > 0:
            removed = original_safe - processed_safe
            print(f"  ✓ Removed {removed} noisy pixels ({removed/original_safe*100:.1f}% reduction)")
    else:
        mask = (prob_map_smoothed >= threshold).astype(np.uint8)
    
    return mask, prob_map_smoothed

def get_feature_importance(clf, feature_names=None):
    if feature_names is None:
        feature_names = ['Elevation', 'Slope', 'Curvature', 'Roughness', 
                        'Mean Slope', 'Max Slope', 'Elev Range']
    
    if hasattr(clf, 'feature_importances_'):
        importance = clf.feature_importances_
        return dict(zip(feature_names, importance))
    return None

def analyze_safe_zones(mask, elevation):
    labeled_array, num_zones = scipy_label(mask)
    
    zones = []
    for zone_id in range(1, num_zones + 1):
        zone_mask = (labeled_array == zone_id)
        size = np.sum(zone_mask)
        
        y_coords, x_coords = np.where(zone_mask)
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        mean_elev = np.mean(elevation[zone_mask])
        
        zones.append({
            'id': zone_id,
            'size': size,
            'mean_elevation': mean_elev,
            'center': (center_x, center_y)
        })
    
    zones.sort(key=lambda x: x['size'], reverse=True)
    
    return zones

if __name__ == "__main__":
    print("This file is not meant to be run directly. Use main.py.")
