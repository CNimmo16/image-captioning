def images_to_patches(images, patch_w, patch_h):
    if images.dim() != 3:
        raise ValueError("Input must be a 3D tensor of shape (batch_size, height, width).")
    
    # Unfold the images into patches
    patches = images.unfold(1, patch_h, patch_w).unfold(2, patch_h, patch_w)
    
    # Reshape to (batch_size, num_patches, patch_h * patch_w)
    batch_size, _, _ = images.shape
    patches = patches.contiguous().view(batch_size, -1, patch_h * patch_w)
    
    return patches
