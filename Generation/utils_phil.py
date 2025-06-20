import re, os
import cv2
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mutual_info_score
from plotly.subplots import make_subplots

from PIL import Image
from skimage import color
from skimage.feature import local_binary_pattern
from scipy.spatial import distance
import random
import torch

def load_and_correct_image(image_path):
    """
    Loads an image, corrects its orientation using EXIF data, converts to RGB format, 
    and center crops it to the largest possible square.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        PIL.Image.Image: A corrected and square-cropped PIL Image object.
    """
    with Image.open(image_path) as image:
        # Correct orientation using EXIF metadata if available
        if hasattr(image, "_getexif") and image._getexif():
            exif = image._getexif()
            orientation_key = 274  # Key for orientation tag
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                # Apply transformations based on orientation value
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        
        # Convert to RGB to ensure consistent processing
        image = image.convert("RGB")
        
        # Perform center square cropping
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image.crop((left, top, right, bottom))
    
    return image

def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for numpy (used in many data science libraries)
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use CUDA, this ensures all GPU seeds are set


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def save_img(np_img, filepath):

    # Make the directory
    os.makedirs('/'.join(filepath.split('/')[:-1]), exist_ok=True)
    
    # Save image
    im = Image.fromarray(np_img)
    im.save(filepath)

# Colors: 
def compute_color_histogram(img):

    hist = {}
    for i, color in enumerate(['r', 'g', 'b']):
        hist[color] = cv2.calcHist([img], [i], None, [256], [0, 256])

    return hist

def compare_colors(img1, img2, measure):

    measures = {'corr': cv2.HISTCMP_CORREL, # higher = better
                'intersect': cv2.HISTCMP_INTERSECT, # higher = better
                'chi2': cv2.HISTCMP_CHISQR, # lower = better
                'bhat': cv2.HISTCMP_BHATTACHARYYA} # lower = better

    hist1 = compute_color_histogram(img1)
    hist2 = compute_color_histogram(img2)

    for color in hist1.keys():
        hist1[color] = cv2.normalize(hist1[color], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        hist2[color] = cv2.normalize(hist2[color], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

    scores = {color: cv2.compareHist(hist1[color], hist2[color], measures[measure]) for color in hist1.keys()}

    return scores


def plot_color_histograms(img1, img2, width = 1600, height = 600, text1 = "Original", text2 = "Generated", measure = 'corr'):
    """Plot color histograms in a 2x3 subplot grid with separate plots for each RGB channel, on the same y-axis scale."""
    colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
    fig = make_subplots(rows=2, cols=3, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=(f"{text1} - Red", f"{text1} - Green", f"{text1} - Blue",
                                        f"{text2} - Red", f"{text2} - Green", f"{text2} - Blue"))
    
    measures = {'corr': cv2.HISTCMP_CORREL, # higher = better
                'intersect': cv2.HISTCMP_INTERSECT, # higher = better
                'chi2': cv2.HISTCMP_CHISQR, # lower = better
                'bhat': cv2.HISTCMP_BHATTACHARYYA} # lower = better    

    # Define RGB fill colors
    fill_rgb = {
        'r': 'rgba(255, 0, 0, 0.5)',   # Red
        'g': 'rgba(0, 255, 0, 0.5)',  # Green
        'b': 'rgba(0, 0, 255, 0.5)',  # Blue
    }

    hist1 = compute_color_histogram(img1)
    hist2 = compute_color_histogram(img2)

    # Find the maximum value across all histograms for uniform y-axis scaling
    max_y_value = max(
        np.max(hist1['r']), np.max(hist1['g']), np.max(hist1['b']),
        np.max(hist2['r']), np.max(hist2['g']), np.max(hist2['b'])
    )

    scores = {}
    for i, color in enumerate(['r', 'g', 'b'], start=1):

        scores[i] = cv2.compareHist(hist1[color], hist2[color], measures[measure])  # Or use cv2.HISTCMP_CHISQR

        # Add histogram for each channel in the original image (top row)
        fig.add_trace(go.Scatter(
            x=list(range(256)),
            y=hist1[color].ravel(),
            mode='lines',
            line=dict(color='black', width=2),
            fill='tozeroy',
            fillcolor=fill_rgb[color],
            name=f'{text1} - {color.upper()} Channel'
        ), row=1, col=i)

        # Add histogram for each channel in the generated image (bottom row)
        fig.add_trace(go.Scatter(
            x=list(range(256)),
            y=hist2[color].ravel(),
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            fill='tozeroy',
            fillcolor=fill_rgb[color],
            name=f'{text2} - {color.upper()} Channel'
        ), row=2, col=i)

        # Set the same y-axis range for all plots
        fig.update_yaxes(range=[0, max_y_value], row=1, col=i)
        fig.update_yaxes(range=[0, max_y_value], row=2, col=i)

    # Update layout with the custom font
    fig.update_layout(
        title=f"Color Histograms for {text1} and {text2} Images by Channel",
        font=dict(
            family="Editorial New Italic",  # Use the exact font name as it appears on your system
            size=14,
        ),
        width=width,  # Adjust width as desired
        height=height   # Adjust height as desired
    )

    # Customize x-axis and y-axis titles for clarity
    for i in range(1, 3+1):
        fig.update_yaxes(title_text="Frequency", row=1, col=i)
        fig.update_yaxes(title_text="Frequency", row=2, col=i)
        fig.update_xaxes(title_text=f"Intensity Value<br>Correlation: {scores[i]:.2f}", row=2, col=i)  # Set only for the second row

    fig.show()


# Textures:

def compute_lbp_histogram(image_np, P=8, R=1):
    """
    Computes LBP and histogram for an image.
    
    Parameters:
    - image_np: numpy array of the image (assumes RGB).
    - P: number of circularly symmetric neighbor set points (typically 8).
    - R: radius of circle (typically 1 for fine details).
    
    Returns:
    - lbp_hist: normalized histogram of LBP values.
    """
    # Convert image to grayscale
    gray_image = color.rgb2gray(image_np)
    
    # Compute LBP
    lbp = local_binary_pattern(gray_image, P, R, method="uniform")
    
    # Compute the histogram of LBP
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return lbp_hist

def compare_textures(img1, img2):

    lbp_hist1 = compute_lbp_histogram(img1)
    lbp_hist2 = compute_lbp_histogram(img2)

    # Compute correlation score
    # TODO: CHECK this
    correlation_score = 1 - distance.correlation(lbp_hist1, lbp_hist2)

    return correlation_score

def plot_lbp_histograms(img1, img2, width=800, height=400, text1="Original", text2="Generated"):

    lbp_hist1 = compute_lbp_histogram(img1)
    lbp_hist2 = compute_lbp_histogram(img2)

    # Compute correlation score
    correlation_score = 1 - distance.correlation(lbp_hist1, lbp_hist2)

    f"""Plot LBP histograms for {text1} and {text2} images."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{text1} LBP Histogram", f"{text2} LBP Histogram"))

    # Find the maximum y-value across both histograms for uniform y-axis scaling
    max_y_value = max(np.max(lbp_hist1), np.max(lbp_hist2))

    # Plot LBP histogram for the original image
    fig.add_trace(go.Scatter(
        x=list(range(len(lbp_hist1))),
        y=lbp_hist1,
        mode='lines',
        line=dict(color='black', width=2),
        fill='tozeroy',
        fillcolor='rgba(192, 192, 192, 0.5)',
        name=f'{text1} LBP'
    ), row=1, col=1)

    # Plot LBP histogram for the generated image
    fig.add_trace(go.Scatter(
        x=list(range(len(lbp_hist2))),
        y=lbp_hist2,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(192, 192, 192, 0.5)',
        name=f'{text2} LBP'
    ), row=1, col=2)

    # Add a dummy trace for displaying the correlation score in the legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],  # Invisible points
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),  # Invisible line
        name=f'Correlation Score: {correlation_score:.2f}'
    ))

    # Set the same y-axis range for both plots
    fig.update_yaxes(range=[0, max_y_value], title="Frequency")

    # Update layout
    fig.update_layout(
        title="Local Binary Patterns Histograms",
        font=dict(
            family="Editorial New Italic",  # Use the exact font name as it appears on your system
            size=14,
        ),
        width=width,
        height=height,
        legend=dict(
            title="Legend",
            font=dict(size=12),
        )
    )

    fig.update_xaxes(title_text="LBP Value", row=1, col=1)
    fig.update_xaxes(title_text="LBP Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)

    fig.show()