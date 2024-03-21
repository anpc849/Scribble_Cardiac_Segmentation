import matplotlib.pyplot as plt

def blend_image_label(image, label, alpha=0.5):
    return (1 - alpha) * image + alpha * label

def plot_all(sample):
    image = sample['image']
    gt = sample['gt']
    label = sample['label']
    blended_image_gt = blend_image_label(image, gt)
    blended_image_scribble = blend_image_label(image, label, 0.3)
    
    plt.figure(figsize=(15, 5))
    # Plot images
    plt.subplot(1, 4, 1)
    plt.imshow(gt, cmap='gray')
    plt.title('gt')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(blended_image_gt, cmap='gray')
    plt.title('img_gt')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(label, cmap='gray')
    plt.title('scribble')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(blended_image_scribble, cmap='gray')
    plt.title('img_scribble')
    plt.axis('off')

    plt.show()
