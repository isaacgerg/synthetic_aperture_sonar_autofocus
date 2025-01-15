#!/usr/bin/env python3
"""
Synthetic Aperture Sonar (SAS) Image Processing Script

This script processes SAS images using both standard and shadow Phase Gradient
Autofocus (PGA). It generates visualization plots combining both analyses and
saves the results as composite images.

Dependencies:
    - numpy
    - matplotlib
    - sas_tools (custom package)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sas_tools
import cv2

def load_test_data(file_path):
    """
    Load the test data from a .npz file.

    Args:
        file_path (str): Path to the .npz file containing test data

    Returns:
        numpy.ndarray: Array of test images
    """
    data = np.load(file_path)
    return data['x_test']

def create_phase_error_plot(phi_hat):
    """
    Generate a plot of phase error vs azimuth bin.

    Args:
        phi_hat (numpy.ndarray): Phase error data

    Returns:
        numpy.ndarray: Processed plot image as a 2D array
    """
    fig = plt.figure()
    plt.plot(phi_hat)
    plt.xlabel('Azimuth Bin')
    plt.ylabel('Phase Error [rad]')
    plt.tight_layout(pad=1.01)

    plot = sas_tools.get_fig_as_numpy(fig)
    plot = np.mean(plot.astype('float32')[:,:,0:3]/255.0, axis=-1)
    plt.close('all')
    return plot

def create_rms_plot(rms):
    """
    Generate a plot of RMS vs iteration.

    Args:
        rms (numpy.ndarray): RMS data

    Returns:
        numpy.ndarray: Processed plot image as a 2D array
    """
    fig = plt.figure()
    plt.plot(rms)
    plt.ylabel('RMS [rad]')
    plt.xlabel('Iteration')
    plt.tight_layout(pad=1.01)

    plot_rms = sas_tools.get_fig_as_numpy(fig)
    plot_rms = np.mean(plot_rms.astype('float32')[:,:,0:3]/255.0, axis=-1)
    plt.close('all')
    return plot_rms

def extract_bright_patch(image, patch_size=64):
    """
    Extract a patch around the brightest point in the image.

    Args:
        image (numpy.ndarray): Input image
        patch_size (int): Size of the square patch to extract

    Returns:
        tuple: (patch, (center_y, center_x)) - The extracted patch and its center coordinates
    """
    # Find brightest point
    max_index = np.unravel_index(np.argmax(np.abs(image)), image.shape)

    # Calculate patch boundaries with padding
    half_size = patch_size // 2
    y_start = max(0, max_index[0] - half_size)
    y_end = min(image.shape[0], max_index[0] + half_size)
    x_start = max(0, max_index[1] - half_size)
    x_end = min(image.shape[1], max_index[1] + half_size)

    # Extract patch
    patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to maintain patch size
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        pad_y_before = half_size - (max_index[0] - y_start)
        pad_y_after = half_size - (y_end - max_index[0])
        pad_x_before = half_size - (max_index[1] - x_start)
        pad_x_after = half_size - (x_end - max_index[1])

        patch = np.pad(patch,
                      ((max(0, pad_y_before), max(0, pad_y_after)),
                       (max(0, pad_x_before), max(0, pad_x_after))),
                      mode='constant')

    return patch, max_index

def process_sas_image(image, shadow_flag):
    """
    Process a single SAS image using PGA.

    Args:
        image (numpy.ndarray): Input SAS image
        shadow_flag (bool): Whether to use shadow PGA

    Returns:
        tuple: (original_drc, processed_drc, phase_plot, rms_plot, original_patch, bright_patch)
    """
    # Apply Schlick tone mapping to input image
    image_drc = sas_tools.schlick(np.abs(image))

    # Extract patch from original image
    original_patch, max_index = extract_bright_patch(image_drc)

    # Apply PGA with custom window parameters
    image_pga, phi_hat, rms = sas_tools.pga(
        image,
        win='custom',
        win_params=[10, 1],
        shadow_pga=shadow_flag
    )

    # Apply Schlick tone mapping to PGA-processed image
    image_drc_pga = sas_tools.schlick(np.abs(image_pga))

    # Extract patch from processed image using same location
    processed_patch = extract_patch_at_location(image_drc_pga, max_index)

    # Generate analysis plots
    phase_plot = create_phase_error_plot(phi_hat)
    rms_plot = create_rms_plot(rms)

    return image_drc, image_drc_pga, phase_plot, rms_plot, original_patch, processed_patch

def create_composite_image(image):
    """
    Process an image with both standard and shadow PGA and create a composite visualization.

    Args:
        image (numpy.ndarray): Input SAS image

    Returns:
        numpy.ndarray: Composite image containing both analyses with labels
    """
    # Process with standard PGA (no shadow)
    image_drc, std_pga, std_phase, std_rms, _, _ = process_sas_image(image, False)

    # Process with shadow PGA
    _, shadow_pga, shadow_phase, shadow_rms, _, _ = process_sas_image(image, True)

    # Create separator line
    separator = np.zeros((512, 1))

    # Stack components for each row
    std_row = np.hstack([
        image_drc,
        separator,
        std_pga,
        separator,
        std_phase,
        separator,
        std_rms
    ])

    shadow_row = np.hstack([
        image_drc,  # Original image
        separator,
        shadow_pga,
        separator,
        shadow_phase,
        separator,
        shadow_rms
    ])

    # Add horizontal separator
    horiz_separator = np.zeros((10, std_row.shape[1]))

    # Combine rows
    composite = np.vstack([
        std_row,
        horiz_separator,
        shadow_row
    ])

    # Convert to uint8 for cv2.putText
    label_img = (composite * 255).astype(np.uint8)

    # Add text labels using cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    color = (255, 255, 255)  # White text

    # Add labels for standard PGA row
    cv2.putText(label_img, 'Standard PGA (ShadowPGA=False)',
                (10, 30), font, font_scale, color, thickness)

    # Add labels for shadow PGA row
    cv2.putText(label_img, 'Shadow PGA (ShadowPGA=True)',
                (10, std_row.shape[0] + horiz_separator.shape[0] + 30),
                font, font_scale, color, thickness)

    # Add column labels
    y_pos = std_row.shape[0] - 20  # Position for column labels
    column_width = image_drc.shape[1]  # Width of each image section

    labels = ['Original', 'PGA Result', 'Phase Error', 'RMS']
    positions = [10,
                column_width + separator.shape[1] + 10,
                2 * (column_width + separator.shape[1]) + 10,
                3 * (column_width + separator.shape[1]) + 10]

    for label, x_pos in zip(labels, positions):
        cv2.putText(label_img, label, (x_pos, y_pos),
                   font, 0.8, color, 1)

    # Convert back to float32 [0,1] range
    return label_img.astype(np.float32) / 255.0

def extract_patch_at_location(image, center_index, patch_size=64):
    """
    Extract a patch around a specified center point.

    Args:
        image (numpy.ndarray): Input image
        center_index (tuple): Center coordinates (y, x)
        patch_size (int): Size of the square patch to extract

    Returns:
        numpy.ndarray: The extracted patch
    """
    half_size = patch_size // 2
    y_start = max(0, center_index[0] - half_size)
    y_end = min(image.shape[0], center_index[0] + half_size)
    x_start = max(0, center_index[1] - half_size)
    x_end = min(image.shape[1], center_index[1] + half_size)

    # Extract patch
    patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to maintain patch size
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        pad_y_before = half_size - (center_index[0] - y_start)
        pad_y_after = half_size - (y_end - center_index[0])
        pad_x_before = half_size - (center_index[1] - x_start)
        pad_x_after = half_size - (x_end - center_index[1])

        patch = np.pad(patch,
                      ((max(0, pad_y_before), max(0, pad_y_after)),
                       (max(0, pad_x_before), max(0, pad_x_after))),
                      mode='constant')

    return patch

def create_detailed_patch_comparison(original_patch, std_patch, shadow_patch):
    """
    Create a detailed comparison of patches with 4x enlargement.

    Args:
        original_patch (numpy.ndarray): Original patch before autofocus
        std_patch (numpy.ndarray): Patch from standard PGA
        shadow_patch (numpy.ndarray): Patch from shadow PGA

    Returns:
        numpy.ndarray: Detailed comparison image
    """
    # Resize patches to 4x their original size using cubic interpolation
    enlarged_orig = cv2.resize(original_patch, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    enlarged_std = cv2.resize(std_patch, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    enlarged_shadow = cv2.resize(shadow_patch, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

    # Create separator
    separator = np.zeros((enlarged_orig.shape[0], 20))

    # Combine patches
    comparison = np.hstack([
        enlarged_orig, separator,
        enlarged_std, separator,
        enlarged_shadow
    ])

    # Convert to uint8 for text
    label_img = (comparison * 255).astype(np.uint8)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)

    # Add title
    title = "64x64 Brightest Region Comparison (4x Enlarged)"
    title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    title_x = (label_img.shape[1] - title_size[0]) // 2
    cv2.putText(label_img, title,
                (title_x, 30), font, font_scale, color, thickness)

    # Add method labels
    orig_x = enlarged_orig.shape[1] // 2 - 60
    std_x = enlarged_orig.shape[1] + separator.shape[1] + enlarged_std.shape[1] // 2 - 60
    shadow_x = enlarged_orig.shape[1] + separator.shape[1] + enlarged_std.shape[1] + \
              separator.shape[1] + enlarged_shadow.shape[1] // 2 - 60

    y_pos = enlarged_orig.shape[0] - 20

    cv2.putText(label_img, "Original",
                (orig_x, y_pos), font, font_scale, color, thickness)
    cv2.putText(label_img, "Standard PGA",
                (std_x, y_pos), font, font_scale, color, thickness)
    cv2.putText(label_img, "Shadow PGA",
                (shadow_x, y_pos), font, font_scale, color, thickness)

    return label_img.astype(np.float32) / 255.0

def main():
    # Configuration
    input_file = r'C:\Users\isaac.gerg\Downloads\test.npz'
    output_dir = 'processed_images'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    x_test = load_test_data(input_file)
    num_images = x_test.shape[0]

    # Process each image
    for img_idx in range(num_images):
        # Extract and process single image
        current_image = x_test[img_idx,:,:]

        # Process with standard PGA
        _, _, _, _, orig_patch1, std_patch = process_sas_image(current_image, False)

        # Process with shadow PGA
        _, _, _, _, orig_patch2, shadow_patch = process_sas_image(current_image, True)

        # Use the same original patch (they should be identical)
        orig_patch = orig_patch1

        # Create main composite image
        composite_image = create_composite_image(current_image)

        # Create detailed patch comparison
        detailed_patches = create_detailed_patch_comparison(
            orig_patch, std_patch, shadow_patch)

        # Save results
        sas_tools.imwrite(composite_image,
                         os.path.join(output_dir, f'image_{img_idx}_combined_pga.png'))
        sas_tools.imwrite(detailed_patches,
                         os.path.join(output_dir, f'image_{img_idx}_patch_comparison.png'))

if __name__ == '__main__':
    main()