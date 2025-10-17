import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label as scipy_label

print("Enhanced visualizer_custom.py loaded successfully!")

def label_gui(image, existing_labels):
    """Interactive Mars terrain labeling GUI"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    im = ax.imshow(image, cmap='terrain', aspect='auto')
    ax.set_title('Mars Terrain Classification - Landing Site Selection\n' +
                'Left Click: SAFE Landing Zone (Green) | Right Click: HAZARD Zone (Red)', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (meters)', rotation=270, labelpad=15)
    
    labeled_points = existing_labels.copy()
    
    for x, y, label in existing_labels:
        if label == 1:
            ax.plot(x, y, 'o', color='lime', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2)
        else:
            ax.plot(x, y, 'X', color='red', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
    
    safe_count = sum(1 for _, _, label in existing_labels if label == 1)
    hazard_count = len(existing_labels) - safe_count
    
    stats_text = ax.text(0.02, 0.98, f'Labels: {safe_count} Safe, {hazard_count} Hazard', 
                        transform=ax.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        verticalalignment='top')
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        label = 1 if event.button == 1 else 0
        labeled_points.append([x, y, label])
        
        if label == 1:
            ax.plot(x, y, 'o', color='lime', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2)
            label_type = 'SAFE ZONE'
        else:
            ax.plot(x, y, 'X', color='red', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
            label_type = 'HAZARD ZONE'
        
        new_safe = sum(1 for _, _, label in labeled_points if label == 1)
        new_hazard = len(labeled_points) - new_safe
        stats_text.set_text(f'Labels: {new_safe} Safe, {new_hazard} Hazard')
        
        fig.canvas.draw()
        print(f"Labeled: ({x}, {y}) -> {label_type}")
    
    def on_save(event):
        plt.close(fig)
    
    def on_clear(event):
        nonlocal labeled_points
        labeled_points = existing_labels.copy()
        ax.clear()
        
        ax.imshow(image, cmap='terrain', aspect='auto')
        ax.set_title('Mars Terrain Classification - Landing Site Selection\n' +
                    'Left Click: SAFE Landing Zone (Green) | Right Click: HAZARD Zone (Red)', 
                    fontsize=14, fontweight='bold')
        
        for x, y, label in existing_labels:
            if label == 1:
                ax.plot(x, y, 'o', color='lime', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2)
            else:
                ax.plot(x, y, 'X', color='red', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        
        stats_text.set_text(f'Labels: {sum(1 for _, _, label in existing_labels if label == 1)} Safe, {len(existing_labels) - sum(1 for _, _, label in existing_labels if label == 1)} Hazard')
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    ax_save = plt.axes([0.85, 0.02, 0.1, 0.05])
    ax_clear = plt.axes([0.74, 0.02, 0.1, 0.05])
    
    btn_save = Button(ax_save, 'Save', color='lightgreen')
    btn_clear = Button(ax_clear, 'Clear', color='lightcoral')
    
    btn_save.on_clicked(on_save)
    btn_clear.on_clicked(on_clear)
    
    plt.tight_layout()
    plt.show()
    
    return labeled_points

def process_probability_map(prob_scores, threshold, smooth_sigma=10, min_area=150):
    """
    Process probability map to create clean, contiguous safe zones.
    """
    prob_smoothed = gaussian_filter(prob_scores, sigma=smooth_sigma)
    prob_smoothed = np.clip(prob_smoothed, 0, 1)
    safe_mask = (prob_smoothed >= threshold).astype(np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    safe_mask = binary_opening(safe_mask, structure=kernel, iterations=3)
    safe_mask = binary_closing(safe_mask, structure=kernel, iterations=2)
    
    labeled_array, num_features = scipy_label(safe_mask)
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_array == region_id)
        if np.sum(region_mask) < min_area:
            safe_mask[region_mask] = 0
    
    overlay = np.zeros_like(prob_smoothed)
    overlay[safe_mask > 0] = prob_smoothed[safe_mask > 0]
    
    return overlay, safe_mask

def plot_results_interactive(elevation, probability_scores, initial_threshold=0.9):
    """Interactive Mars landing site visualization with overlay"""
    
    colors = ['darkred', 'red', 'orange', 'yellow', 'yellowgreen', 'limegreen', 'green', 'darkgreen']
    n_bins = 256
    safety_cmap = LinearSegmentedColormap.from_list('safety', colors, N=n_bins)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.05, 1, 0.1], width_ratios=[1, 1])
    fig.suptitle('Mars Landing Site Analysis - Interactive Safety Assessment', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(elevation, cmap='terrain', aspect='auto')
    ax1.set_title('Mars Digital Elevation Model', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (pixels)')
    ax1.set_ylabel('Latitude (pixels)')
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Elevation (meters)', rotation=270, labelpad=15)
    ax2 = fig.add_subplot(gs[1, 1])
    terrain_bg = ax2.imshow(elevation, cmap='terrain', aspect='auto', alpha=0.4)
    current_threshold = initial_threshold
    overlay, safe_mask = process_probability_map(probability_scores, current_threshold, 
                                                  smooth_sigma=10, min_area=150)
    
    im2 = ax2.imshow(overlay, cmap=safety_cmap, aspect='auto', alpha=0.9, 
                     vmin=current_threshold, vmax=1.0, interpolation='bilinear')
    ax2.set_title(f'Landing Safety Overlay (Threshold: {current_threshold:.2f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude (pixels)')
    ax2.set_ylabel('Latitude (pixels)')
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Landing Safety Probability', rotation=270, labelpad=15)
    
    ax_slider = fig.add_subplot(gs[2, :])
    slider = Slider(ax_slider, 'Safety Threshold', 0.1, 0.99, 
                   valinit=current_threshold, valfmt='%.2f')
    
    threshold_text = fig.text(0.5, 0.12, get_threshold_description(current_threshold), 
                             ha='center', fontsize=11, style='italic')

    safe_percentage = (safe_mask.sum() / safe_mask.size) * 100
    
    labeled_zones, num_zones = scipy_label(safe_mask)
    
    stats_text = fig.text(0.02, 0.88, 
                         f'📊 Analysis Results:\n' +
                         f'• Safe Landing Area: {safe_percentage:.1f}%\n' +
                         f'• Distinct Landing Zones: {num_zones}\n' +
                         f'• Threshold Level: {current_threshold:.2f}\n' +
                         f'• Total Terrain: {elevation.size:,} pixels',
                         fontsize=10, 
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    def update_threshold(val):
        nonlocal current_threshold
        current_threshold = slider.val
        overlay, safe_mask = process_probability_map(probability_scores, current_threshold,
                                                      smooth_sigma=10, min_area=150)
    
        im2.set_array(overlay)
        im2.set_clim(vmin=current_threshold, vmax=1.0)
        ax2.set_title(f'Landing Safety Overlay (Threshold: {current_threshold:.2f})', 
                      fontsize=12, fontweight='bold')
    
        safe_percentage = (safe_mask.sum() / safe_mask.size) * 100
        labeled_zones, num_zones = scipy_label(safe_mask)
        
        stats_text.set_text(
            f'📊 Analysis Results:\n' +
            f'• Safe Landing Area: {safe_percentage:.1f}%\n' +
            f'• Distinct Landing Zones: {num_zones}\n' +
            f'• Threshold Level: {current_threshold:.2f}\n' +
            f'• Total Terrain: {elevation.size:,} pixels'
        )

        threshold_text.set_text(get_threshold_description(current_threshold))
        fig.canvas.draw()
    
    slider.on_changed(update_threshold)
    
    legend_elements = [
        patches.Patch(color='green', alpha=0.8, label='Safe Landing Zones'),
        patches.Patch(color='yellow', alpha=0.8, label='Marginal Areas'),
        patches.Patch(color='red', alpha=0.8, label='Hazardous Terrain')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.show()

def get_threshold_description(threshold):
    """Get human-readable description of threshold setting"""
    if threshold >= 0.9:
        return "VERY STRICT: Only highest-confidence safe areas (Mission Critical)"
    elif threshold >= 0.8:
        return "STRICT: High-confidence safe areas (Recommended for landing)"
    elif threshold >= 0.6:
        return "MODERATE: Moderate-confidence areas (Acceptable risk)"
    elif threshold >= 0.4:
        return "LENIENT: Lower-confidence areas (Higher risk tolerance)"
    else:
        return "VERY LENIENT: All potentially safe areas (Maximum coverage)"

def plot_results(elevation, safe_mask):
    """Backward compatibility function"""
    prob_scores = safe_mask.astype(float)
    plot_results_interactive(elevation, prob_scores, initial_threshold=0.5)
