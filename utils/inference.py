import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
def inference(dataset, idx, model, device):
    image = dataset[idx]['image'].unsqueeze(0)

    model.eval()

    with torch.no_grad():
        preds = torch.argmax(F.softmax(model(image.to(device))[0], dim=1), dim=1)
        mask_img = preds.squeeze(0)

    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze().cpu().detach().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_img.cpu().detach().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(dataset[idx]['gt'].cpu(), cmap='gray')
    plt.axis('off')
    plt.show()