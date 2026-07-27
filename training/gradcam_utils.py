"""
Grad-CAM Utilities for Potato Disease Classification

This module provides:
1. Grad-CAM visualization for model interpretability
2. Disease region masking from heatmaps
3. Disease severity quantification metrics

Usage in notebooks:
    from gradcam_utils import GradCAMAnalyzer
    
    analyzer = GradCAMAnalyzer(model, class_names=["Early Blight", "Late Blight", "Healthy"])
    results = analyzer.analyze(image)
    analyzer.visualize(results)
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalysisResult:
    """Container for Grad-CAM analysis results."""
    image: np.ndarray
    heatmap_raw: np.ndarray
    heatmap_overlay: np.ndarray
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    mask: np.ndarray
    severity: Dict[str, float]
    masked_image: np.ndarray


class GradCAMAnalyzer:
    """
    Grad-CAM analyzer for potato disease classification.
    
    Provides visualization, disease region masking, and severity quantification.
    """
    
    # Convolutional layer types to search for
    CONV_LAYER_TYPES = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )
    
    # Severity thresholds (percentage of affected area)
    SEVERITY_THRESHOLDS = {
        'healthy': (0, 8),        # 0-8% affected = healthy
        'mild': (8, 20),          # 8-20% affected = mild
        'moderate': (20, 50),     # 20-50% affected = moderate
        'severe': (50, 100),      # 50-100% affected = severe
    }
    
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: List[str],
        image_size: int = 256,
        mask_threshold: float = 0.5,
        heatmap_threshold: float = 0.4
    ):
        """
        Initialize the Grad-CAM analyzer.
        
        Args:
            model: Trained Keras model
            class_names: List of class names (e.g., ["Early Blight", "Late Blight", "Healthy"])
            image_size: Expected input image size
            mask_threshold: Threshold for binary mask (0-1)
            heatmap_threshold: Threshold for heatmap visualization (0-1)
        """
        self.model = model
        self.class_names = class_names
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.heatmap_threshold = heatmap_threshold
        self._conv_layer_cache = {}
    
    def _find_last_conv_layer(self, model_id: str = "default") -> Tuple[str, tf.keras.layers.Layer]:
        """
        Find the last convolutional layer suitable for Grad-CAM.
        
        For standard models, finds the deepest Conv2D layer.
        For MobileNetV2-like models, skips depthwise layers and finds
        Conv2D layers with spatial dims >= 14x14.
        
        Returns:
            Tuple of (layer_name, layer_object)
        """
        if model_id in self._conv_layer_cache:
            return self._conv_layer_cache[model_id]
        
        # MobileNetV2-specific: skip depthwise, target Conv2D >= 14x14
        if hasattr(self.model, '_name') and 'mobilenet' in self.model._name.lower():
            candidates = []
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    shape = layer.output_shape
                    if len(shape) == 4:
                        h, w = shape[1], shape[2]
                        if h is not None and w is not None and h >= 14 and w >= 14:
                            candidates.append((layer.name, layer, h * w))
            if candidates:
                candidates.sort(key=lambda c: c[2])  # smallest area = deepest with >=14x14
                best = candidates[0]
                self._conv_layer_cache[model_id] = (best[0], best[1])
                return (best[0], best[1])
        
        # Default: find deepest conv layer
        best = [None, None]
        best_depth = [-1]
        
        def search(layer, depth=0, containing=None):
            if isinstance(layer, self.CONV_LAYER_TYPES):
                if depth >= best_depth[0]:
                    best[0] = layer.name
                    best[1] = containing or layer
                    best_depth[0] = depth
            elif hasattr(layer, 'layers'):
                for sub in layer.layers:
                    search(sub, depth + 1, containing=layer)
        
        for layer in self.model.layers:
            search(layer, 0, containing=None)
        
        result = (best[0], best[1])
        self._conv_layer_cache[model_id] = result
        return result
    
    def _find_layer_in_model(self, model: tf.keras.Model, layer_name: str) -> Optional[tf.keras.layers.Layer]:
        """Recursively find a layer by name in the model graph."""
        for layer in model.layers:
            if layer.name == layer_name:
                return layer
            if hasattr(layer, 'layers'):
                result = self._find_layer_in_model(layer, layer_name)
                if result is not None:
                    return result
        return None
    
    def compute_gradcam(
        self,
        image: np.ndarray,
        model_id: str = "default"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute Grad-CAM heatmap for the given image.
        
        Args:
            image: Input image (H, W, 3), values 0-255
            model_id: Model identifier for caching
            
        Returns:
            Tuple of (heatmap, predictions_dict)
        """
        # Preprocess image
        img_tensor = tf.cast(np.expand_dims(image, 0), tf.float32)
        
        # Find conv layer
        layer_name, _ = self._find_last_conv_layer(model_id)
        actual_layer = self._find_layer_in_model(self.model, layer_name)
        
        if actual_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model graph")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[actual_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        
        # Get predictions
        pred_array = predictions[0].numpy()
        probs = {name: float(pred_array[i]) for i, name in enumerate(self.class_names)}
        
        return heatmap.numpy(), probs
    
    def create_disease_mask(
        self,
        heatmap: np.ndarray,
        threshold: Optional[float] = None,
        method: str = "threshold"
    ) -> np.ndarray:
        """
        Create a binary mask from the heatmap to isolate disease regions.
        
        Args:
            heatmap: Raw heatmap (H, W)
            threshold: Threshold for binarization (0-1)
            method: Masking method - "threshold", "otsu", or "adaptive"
            
        Returns:
            Binary mask (H, W) with values 0 or 255
        """
        if threshold is None:
            threshold = self.mask_threshold
        
        # Resize heatmap to image size
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis], 
            (self.image_size, self.image_size)
        ).numpy().squeeze()
        
        if method == "threshold":
            # Simple thresholding
            mask = (heatmap_resized > threshold).astype(np.uint8) * 255
            
        elif method == "otsu":
            # Otsu's method (manual implementation)
            hist, bin_edges = np.histogram(heatmap_resized.ravel(), bins=256, range=(0, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Compute Otsu's threshold
            total = hist.sum()
            sum_total = np.sum(hist * bin_centers)
            sum_bg = 0
            weight_bg = 0
            max_variance = 0
            best_threshold = 0
            
            for i in range(256):
                weight_bg += hist[i]
                if weight_bg == 0:
                    continue
                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break
                sum_bg += hist[i] * bin_centers[i]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                if variance > max_variance:
                    max_variance = variance
                    best_threshold = bin_centers[i]
            
            mask = (heatmap_resized > best_threshold).astype(np.uint8) * 255
            
        elif method == "adaptive":
            # Adaptive thresholding based on local mean
            local_mean = ndimage.uniform_filter(heatmap_resized, size=15)
            mask = (heatmap_resized > local_mean * 0.8).astype(np.uint8) * 255
            
        else:
            raise ValueError(f"Unknown masking method: {method}")
        
        # Apply morphological operations to clean up mask
        binary_mask = mask > 0
        binary_mask = ndimage.binary_opening(binary_mask, iterations=2)
        binary_mask = ndimage.binary_closing(binary_mask, iterations=2)
        
        return binary_mask.astype(np.uint8) * 255
    
    def quantify_severity(
        self,
        mask: np.ndarray,
        heatmap: np.ndarray,
        predicted_class: str
    ) -> Dict[str, float]:
        """
        Quantify disease severity based on the mask and heatmap.
        
        Args:
            mask: Binary disease mask (H, W)
            heatmap: Raw heatmap (H, W)
            predicted_class: Predicted disease class
            
        Returns:
            Dictionary with severity metrics
        """
        # Resize heatmap if needed
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            (self.image_size, self.image_size)
        ).numpy().squeeze()
        
        # Calculate affected area percentage
        total_pixels = mask.shape[0] * mask.shape[1]
        affected_pixels = np.sum(mask > 0)
        affected_percentage = (affected_pixels / total_pixels) * 100
        
        # Calculate mean intensity in affected regions
        if affected_pixels > 0:
            mean_intensity = float(np.mean(heatmap_resized[mask > 0]))
            max_intensity = float(np.max(heatmap_resized[mask > 0]))
        else:
            mean_intensity = 0.0
            max_intensity = 0.0
            severity_level = "healthy"
        
        # Determine severity level
        severity_level = "unknown"
        for level, (low, high) in self.SEVERITY_THRESHOLDS.items():
            if low <= affected_percentage < high:
                severity_level = level
                break
        
        # For healthy plants, low affected percentage is expected
        if predicted_class.lower() == "healthy":
            if affected_percentage < 5:
                severity_level = "healthy"
            else:
                severity_level = "false_positive"
        
        # Calculate intensity-weighted severity
        # This considers both area and how intense the disease signal is
        intensity_weighted_severity = affected_percentage * mean_intensity
        
        return {
            "affected_percentage": round(affected_percentage, 2),
            "mean_intensity": round(mean_intensity, 4),
            "max_intensity": round(max_intensity, 4),
            "severity_level": severity_level,
            "intensity_weighted_severity": round(intensity_weighted_severity, 2),
            "total_pixels": total_pixels,
            "affected_pixels": int(affected_pixels),
        }
    
    def create_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create a heatmap overlay on the original image.
        
        Args:
            image: Original image (H, W, 3)
            heatmap: Raw heatmap (H, W)
            alpha: Blending factor
            
        Returns:
            Overlay image (H, W, 3)
        """
        # Resize heatmap
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            (self.image_size, self.image_size)
        ).numpy().squeeze()
        
        # Apply colormap
        heatmap_normalized = np.clip(heatmap_resized, 0, 1)
        heatmap_gamma = np.power(heatmap_normalized, 0.7)
        colored = cm.jet(heatmap_gamma)[:, :, :3]
        colored_255 = (colored * 255).astype(np.uint8)
        
        # Blend
        overlay = (image.astype(np.float32) * (1 - alpha) + 
                   colored_255.astype(np.float32) * alpha)
        
        return np.clip(overlay, 0, 255).astype(np.uint8)
    
    def create_masked_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Create an image with disease regions highlighted and rest masked.
        
        Args:
            image: Original image (H, W, 3)
            mask: Binary mask (H, W)
            background_color: Color for masked-out regions
            
        Returns:
            Masked image (H, W, 3)
        """
        masked = image.copy()
        mask_binary = mask > 0
        
        # Set non-disease regions to background color
        masked[~mask_binary] = background_color
        
        # Add red border around disease regions for visibility
        border = ndimage.binary_dilation(mask_binary, iterations=3) & ~mask_binary
        masked[border] = [255, 0, 0]  # Red border
        
        return masked.astype(np.uint8)
    
    def analyze(
        self,
        image: np.ndarray,
        model_id: str = "default",
        include_mask: bool = True,
        include_severity: bool = True
    ) -> AnalysisResult:
        """
        Perform complete analysis on an image.
        
        Args:
            image: Input image (H, W, 3), values 0-255
            model_id: Model identifier for caching
            include_mask: Whether to compute disease mask
            include_severity: Whether to compute severity metrics
            
        Returns:
            AnalysisResult with all computed metrics
        """
        # Resize image if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = np.array(
                Image.fromarray(image).resize(
                    (self.image_size, self.image_size), 
                    Image.Resampling.LANCZOS
                )
            )
        
        # Compute Grad-CAM
        heatmap, probabilities = self.compute_gradcam(image, model_id)
        
        # Get prediction
        prediction = max(probabilities, key=probabilities.get)
        confidence = probabilities[prediction]
        
        # Create overlay
        overlay = self.create_overlay(image, heatmap)
        
        # Compute mask and severity if requested
        if include_mask:
            mask = self.create_disease_mask(heatmap)
            masked_image = self.create_masked_image(image, mask)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            masked_image = image.copy()
        
        if include_severity:
            severity = self.quantify_severity(mask, heatmap, prediction)
        else:
            severity = {}
        
        return AnalysisResult(
            image=image,
            heatmap_raw=heatmap,
            heatmap_overlay=overlay,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            mask=mask,
            severity=severity,
            masked_image=masked_image
        )
    
    def visualize(
        self,
        result: AnalysisResult,
        figsize: Tuple[int, int] = (16, 10),
        show_severity: bool = True
    ):
        """
        Visualize the analysis results.
        
        Args:
            result: AnalysisResult from analyze()
            figsize: Figure size
            show_severity: Whether to show severity metrics
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            f"Prediction: {result.prediction} ({result.confidence:.1%})",
            fontsize=16, fontweight='bold'
        )
        
        # Row 1: Original, Heatmap Overlay, Raw Heatmap
        axes[0, 0].imshow(result.image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(result.heatmap_overlay)
        axes[0, 1].set_title("Grad-CAM Overlay")
        axes[0, 1].axis('off')
        
        # Raw heatmap
        heatmap_resized = tf.image.resize(
            result.heatmap_raw[..., np.newaxis],
            (self.image_size, self.image_size)
        ).numpy().squeeze()
        im = axes[0, 2].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        axes[0, 2].set_title("Raw Heatmap")
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Row 2: Mask, Masked Image, Severity
        axes[1, 0].imshow(result.mask, cmap='gray')
        axes[1, 0].set_title("Disease Mask")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(result.masked_image)
        axes[1, 1].set_title("Isolated Disease Regions")
        axes[1, 1].axis('off')
        
        # Severity info
        axes[1, 2].axis('off')
        if show_severity and result.severity:
            severity_text = (
                f"Severity Analysis\n"
                f"{'─' * 25}\n"
                f"Affected Area: {result.severity['affected_percentage']:.1f}%\n"
                f"Mean Intensity: {result.severity['mean_intensity']:.3f}\n"
                f"Max Intensity: {result.severity['max_intensity']:.3f}\n"
                f"Severity Level: {result.severity['severity_level'].upper()}\n"
                f"{'─' * 25}\n"
                f"Probabilities:\n"
            )
            for cls, prob in result.probabilities.items():
                bar = '█' * int(prob * 20)
                severity_text += f"  {cls}: {prob:.1%} {bar}\n"
            
            axes[1, 2].text(
                0.1, 0.9, severity_text,
                transform=axes[1, 2].transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        plt.show()
    
    def visualize_comparison(
        self,
        results: List[AnalysisResult],
        titles: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (18, 6)
    ):
        """
        Compare multiple analysis results side by side.
        
        Args:
            results: List of AnalysisResult objects
            titles: Optional titles for each result
            figsize: Figure size
        """
        n = len(results)
        fig, axes = plt.subplots(2, n, figsize=figsize)
        if n == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle("Disease Analysis Comparison", fontsize=16, fontweight='bold')
        
        for i, result in enumerate(results):
            title = titles[i] if titles else f"Sample {i + 1}"
            
            # Top row: Original + Overlay
            axes[0, i].imshow(result.heatmap_overlay)
            axes[0, i].set_title(f"{title}\n{result.prediction} ({result.confidence:.1%})")
            axes[0, i].axis('off')
            
            # Bottom row: Masked image
            axes[1, i].imshow(result.masked_image)
            severity = result.severity.get('severity_level', 'N/A')
            affected = result.severity.get('affected_percentage', 0)
            axes[1, i].set_title(f"Severity: {severity}\nAffected: {affected:.1f}%")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


def load_and_analyze(
    model_path: str,
    image_path: str,
    class_names: List[str] = ["Early Blight", "Late Blight", "Healthy"],
    image_size: int = 256,
    visualize: bool = True
) -> AnalysisResult:
    """
    Convenience function to load a model and analyze an image.
    
    Args:
        model_path: Path to saved model
        image_path: Path to input image
        class_names: List of class names
        image_size: Expected input image size
        visualize: Whether to display results
        
    Returns:
        AnalysisResult
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create analyzer
    analyzer = GradCAMAnalyzer(
        model=model,
        class_names=class_names,
        image_size=image_size
    )
    
    # Load image
    image = np.array(
        Image.open(image_path).convert('RGB').resize(
            (image_size, image_size),
            Image.Resampling.LANCZOS
        )
    )
    
    # Analyze
    result = analyzer.analyze(image)
    
    # Visualize
    if visualize:
        analyzer.visualize(result)
    
    return result


# Example usage for notebooks
def notebook_demo():
    """
    Demo function showing how to use the analyzer in a notebook.
    """
    demo_code = """
    # 1. Import the analyzer
    from gradcam_utils import GradCAMAnalyzer
    
    # 2. Load your model
    import tensorflow as tf
    model = tf.keras.models.load_model("../saved_models/1")  # CNN Baseline
    
    # 3. Create analyzer
    analyzer = GradCAMAnalyzer(
        model=model,
        class_names=["Early Blight", "Late Blight", "Healthy"]
    )
    
    # 4. Load and preprocess image
    from PIL import Image
    import numpy as np
    
    image_path = "path/to/your/potato_leaf.jpg"
    image = np.array(
        Image.open(image_path).convert('RGB').resize((256, 256))
    )
    
    # 5. Run analysis
    result = analyzer.analyze(image)
    
    # 6. Visualize results
    analyzer.visualize(result)
    
    # 7. Access severity metrics
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Severity: {result.severity['severity_level']}")
    print(f"Affected Area: {result.severity['affected_percentage']:.1f}%")
    
    # 8. Compare multiple images
    results = [analyzer.analyze(img1), analyzer.analyze(img2), analyzer.analyze(img3)]
    analyzer.visualize_comparison(results, titles=["Healthy", "Early Blight", "Late Blight"])
    """
    print(demo_code)


if __name__ == "__main__":
    print("Grad-CAM Utilities for Potato Disease Classification")
    print("=" * 55)
    print("\nThis module provides:")
    print("  1. Grad-CAM visualization")
    print("  2. Disease region masking")
    print("  3. Disease severity quantification")
    print("\nUsage: from gradcam_utils import GradCAMAnalyzer")
    print("\nRunning demo code...")
    notebook_demo()
