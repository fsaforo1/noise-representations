import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable


def vae_compare_samples(model, test_loader, batch_size):
    # Compare original and reconstructed samples
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.float()
            if torch.cuda.is_available():
                data = data.cuda()
            recon_batch, mu, logvar, _ = model(data)
            if i == 1:
                n = min(data.size(0), 8)
                orig = data[:n]
                recon = recon_batch.view(batch_size, 3, 32, 32)[:n]
                fig = plt.figure(figsize=(8, 8))
                columns = 4
                rows = 4
                combined = torch.cat([orig.cpu(), recon.cpu()])
                for i in range(1, 2 * n + 1):
                    if i % 2 == 1:
                        img = combined[int(i / 2)][:, :, :].transpose(0, 2).transpose(0, 1)
                    else:
                        img = combined[int(i / 2) + n - 1][:, :, :].transpose(0, 2).transpose(0, 1)
                    fig.add_subplot(rows, columns, i)
                    if i % 2 == 1:
                        plt.title("Original: " + str(int(i / 2) + 1))
                    else:
                        plt.title("Reconstructed: " + str(int(i / 2)))
                    plt.axis('off')
                    plt.imshow(img, cmap='gray')
                plt.show()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
    return BCE + KLD


def train_vae(model, optimizer, train_loader, test_loader, batch_size, num_epochs, add_noise=False, noise_size=0.2):
    val_batches = len(test_loader)
    train_losses = np.zeros(num_epochs).astype(float)
    test_losses = np.zeros(num_epochs).astype(float)
    print("Epoch: ", end="")
    for epoch in range(num_epochs):
        print(epoch, end=" ")
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.float())
            if torch.cuda.is_available():
                images = images.cuda()
            if add_noise:
                noise = torch.rand(batch_size, 3, 32, 32)
                if torch.cuda.is_available():
                    noise = noise.cuda()
                noise = noise * noise_size - noise_size * 0.5
                images_noise = images + noise

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            if add_noise:
                recon_batch, mu, logvar, _ = model(images_noise)
            else:
                recon_batch, mu, logvar, _ = model(images)
            loss = loss_function(recon_batch, images, mu, logvar)
            loss.backward()
            optimizer.step()
        train_losses[epoch] = loss.data

        model.eval()
        val_losses = 0

        for i, (images, labels) in enumerate(test_loader):
            images_test = Variable(images.float())
            if torch.cuda.is_available():
                images_test = images_test.cuda()
            recon_batch, mu, logvar, _ = model(images_test)

            val_losses += loss_function(recon_batch, images_test, mu, logvar).item()
        print('Epoch : %d/%d, Test Loss: %.4f' % (epoch + 1, num_epochs, 1.0 * val_losses / val_batches))
        test_losses[epoch] = 1.0 * val_losses / val_batches
    return train_losses, test_losses


def graph_loss(train_losses, test_losses, plot_name, num_epochs):
    x = np.arange(num_epochs)
    plt.plot(x, train_losses, label='Train')
    plt.plot(x, test_losses, label='Test')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss value")
    plt.legend()
    plt.savefig(plot_name + ".png")
    plt.show()


def graph_norms(model, testset, test_loader, batch_size):
    # Generate set of all 2-norms of encoding vectors
    size = len(testset)
    norms = np.zeros(size, dtype=float)
    norms_z = norms.copy()
    model.eval()
    counter = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images_test = Variable(images.float())
            if torch.cuda.is_available():
                images_test = images_test.cuda()
            recon_batch, mu, logvar, z = model(images_test)
            for j in range(batch_size):
                encoding = mu[j]
                encoding_reg = z[j]
                norms[counter] = torch.sum(((encoding - encoding.mean()) / encoding.std()).pow(2)).pow(0.5)
                norms_z[counter] = torch.norm(encoding_reg, p=2, dim=0)
                if i == 0 and j == 0:
                    print(mu.shape, batch_size, norms[counter], norms_z[counter])
                counter += 1
    # Display the distribution of 2-norms for encoding vectors (mu) and regularized encoding vectors
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    ax_0 = plt.subplot('121')
    ax_0.hist(norms, bins=8)
    ax_0.set_title('Mu encoding distribution')

    ax_1 = plt.subplot('122')
    ax_1.hist(norms_z, bins=50)
    ax_1.set_title('Regularized encoding distribution')
    plt.show()
