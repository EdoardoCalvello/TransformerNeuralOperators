import torch
import torch.optim as optim
# project-specific imports
import ViT
from utils_vit import get_dataloader, train_and_eval, plot_losses

DEBUG=False

if __name__ == "__main__":

    # Define hyperparameters
    image_size = 224
    patch_size = 16
    num_classes = 10
    dim = 128
    encoder_depth = 3
    num_heads = 4
    mlp_dim = 256
    dropout=0.1
    batch_size = 64
    num_epochs = 10

    # Load data
    train_loader = get_dataloader(image_size, batch_size, split='train')
    test_loader = get_dataloader(image_size, batch_size, split='test')

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT(image_size, patch_size, num_classes, dim, encoder_depth, num_heads, mlp_dim, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if DEBUG:
        print(device)

    # Get number of trainable parameters
    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if DEBUG:
        print("This architecture has {} trainable parameters.".format(trainable_params))


    train_loss_list, test_loss_list = train_and_eval(model,
                                                    optimizer,
                                                    device,
                                                    train_loader,
                                                    test_loader,
                                                    num_epochs,
                                                    DEBUG=DEBUG)

    # Plot loss vs number of epochs
    plot_losses(train_loss_list, test_loss_list)
