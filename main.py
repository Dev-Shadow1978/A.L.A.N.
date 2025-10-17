import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import core_logic_custom as core
import visualizer_custom as viz

DEM_PATH = "[YOUR_CUSTOM_PATH]/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
LABEL_PATH = "labels/labeled_points.json"
MODEL_PATH = "models/classifier_model.npz"
PREDICTION_THRESHOLD = 0.9
MIN_LANDING_ZONE_SIZE = 100

def ensure_directories():
    os.makedirs("labels", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def load_existing_model():
    if os.path.exists(MODEL_PATH):
        try:
            data = np.load(MODEL_PATH, allow_pickle=True)
            print(f"✓ Loaded existing model from {MODEL_PATH}")
            return {
                'clf': data['clf'].item(),
                'data': (data['X'], data['Y'])
            }
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
    return None

def save_model(clf, X, Y):
    """Save trained model."""
    try:
        np.savez(MODEL_PATH, clf=clf, X=X, Y=Y)
        print(f"✓ Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Could not save model: {e}")

def main():
    print("=" * 70)
    print("🚀 MARS LANDING SITE CLASSIFICATION SYSTEM - ENHANCED")
    print("=" * 70)
    
    ensure_directories()

    if not os.path.exists(DEM_PATH):
        print(f"\n❌ ERROR: DEM file not found at {DEM_PATH}")
        print("Please update DEM_PATH in main.py to point to your Mars DEM file.")
        return
    
    print(f"\n📡 Loading DEM from {DEM_PATH}...")
    try:
        elevation = core.load_dem(DEM_PATH)
        print(f"✓ DEM loaded successfully!")
        print(f"  Shape: {elevation.shape}")
        print(f"  Elevation range: {np.nanmin(elevation):.1f} to {np.nanmax(elevation):.1f} meters")
    except Exception as e:
        print(f"❌ Failed to load DEM: {e}")
        return
    
    print("\n🔬 Computing enhanced terrain features (including spatial context)...")
    try:
        features_dict = core.compute_features(elevation)
        print("✓ Enhanced features computed successfully!")
        print(f"  • Elevation range: {np.nanmin(features_dict['elevation']):.1f} to {np.nanmax(features_dict['elevation']):.1f} m")
        print(f"  • Slope range: {np.nanmin(features_dict['slope']):.3f} to {np.nanmax(features_dict['slope']):.3f}")
        print(f"  • Terrain roughness: {np.nanmin(features_dict['roughness']):.3f} to {np.nanmax(features_dict['roughness']):.3f}")
        print(f"  • Local slope variation computed over {core.WINDOW_SIZE}x{core.WINDOW_SIZE} pixel windows")
    except Exception as e:
        print(f"❌ Failed to compute features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n📋 Loading labels from {LABEL_PATH}...")
    existing_labels = core.load_labels(LABEL_PATH)
    print(f"✓ Found {len(existing_labels)} existing labels")
    
    print("\n🎨 Opening interactive labeling interface...")
    print("Instructions:")
    print("  • LEFT CLICK: Mark as SAFE landing zone (green)")
    print("  • RIGHT CLICK: Mark as HAZARD zone (red)")
    print("  • Click 'Save' button when done")
    print("  • Click 'Clear' to remove new labels")
    print("\n💡 TIP: Label entire flat regions, not just individual pixels!")
    
    try:
        updated_labels = viz.label_gui(elevation, existing_labels)
        
        if len(updated_labels) > len(existing_labels):
            print(f"\n✓ Added {len(updated_labels) - len(existing_labels)} new labels")
            core.save_labels(LABEL_PATH, updated_labels)
        else:
            print("\n⚠ No new labels added")
        
        all_labels = updated_labels
        
    except Exception as e:
        print(f"❌ Error in labeling interface: {e}")
        print("Using existing labels only...")
        all_labels = existing_labels
    
    if len(all_labels) < 2:
        print("\n⚠ Not enough labels to train a model (need at least 2)")
        print("Please add more labels and run again.")
        return
    
    print("\n🔧 Preparing training data with spatial features...")
    try:
        X, Y, feature_names = core.build_training_data(features_dict, all_labels)
        print(f"✓ Training data prepared: {len(X)} samples with {X.shape[1]} features")
        
        safe_count = np.sum(Y == 1)
        hazard_count = np.sum(Y == 0)
        print(f"  • Safe zones: {safe_count}")
        print(f"  • Hazard zones: {hazard_count}")
        
    except Exception as e:
        print(f"❌ Failed to prepare training data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    existing_model = load_existing_model()
    
    print("\n🤖 Training enhanced Random Forest classifier (200 trees)...")
    try:
        clf, training_data = core.train_model(X, Y, existing_model)
        print("✓ Model trained successfully!")
        
        importance = core.get_feature_importance(clf, feature_names)
        if importance:
            print("\n📊 Feature Importance (Top contributors):")
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, value in sorted_features[:5]:
                bar = "█" * int(value * 50)
                print(f"  • {feature:15s}: {bar} {value:.3f}")
        
        save_model(clf, *training_data)
        
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎯 Generating predictions with spatial post-processing...")
    print(f"  • Safety threshold: {PREDICTION_THRESHOLD}")
    print(f"  • Minimum landing zone size: {MIN_LANDING_ZONE_SIZE} pixels")
    
    try:
        safe_mask, prob_scores = core.predict_safe_areas(
            clf, 
            features_dict, 
            threshold=PREDICTION_THRESHOLD,
            apply_postprocess=True,
            min_area=MIN_LANDING_ZONE_SIZE
        )
        
        safe_percentage = safe_mask.sum() / safe_mask.size * 100
        print(f"✓ Predictions generated!")
        print(f"  • Safe landing area: {safe_percentage:.1f}% of terrain")
        
        zones = core.analyze_safe_zones(safe_mask, elevation)
        if zones:
            print(f"  • Identified {len(zones)} distinct safe landing zones")
            print(f"\n🎯 Top 3 Landing Zones:")
            for i, zone in enumerate(zones[:3], 1):
                print(f"    {i}. Zone {zone['id']}: {zone['size']} pixels "
                      f"at elevation {zone['mean_elevation']:.1f}m "
                      f"(center: {zone['center']})")
        else:
            print("  • No safe landing zones found at current threshold")
        
    except Exception as e:
        print(f"❌ Failed to generate predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n📈 Opening interactive results visualization...")
    print("Features:")
    print("  • Drag slider to adjust safety threshold")
    print("  • See contiguous safe landing zones")
    print("  • Spatial smoothing removes noise")
    
    try:
        viz.plot_results_interactive(elevation, prob_scores, initial_threshold=PREDICTION_THRESHOLD)
        print("\n✓ Session complete!")
        
    except Exception as e:
        print(f"❌ Error showing results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Thank you for using the Enhanced Mars Landing Site Classification System!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
