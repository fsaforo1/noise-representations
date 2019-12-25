import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import random

from CNN_aux_functions import run_cuda_model
from random_noise import random_noise
from vae import VAE
from VAE_aux_functions import vae_compare_samples, train_vae, graph_loss, graph_norms


if __name__ == '__main__':


    def download_file_from_google_drive(id, destination):
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)


    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None


    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    # resnet_local.pt https://drive.google.com/open?id=1IMRxCoRjgnc7bqyu16qtMYFRgVDfsTaL
    # resnet_poisson https://drive.google.com/open?id=13lOdLsLqpgRXpIY6-XorkDhQi8xT5u_I
    # resnet_gaussian https://drive.google.com/open?id=1jZi37FXyUd1zq0AC8ERTDg7YFocNkPrX
    # resnet_blots https://drive.google.com/open?id=1ohdXmA-7SKNjzVa8SRWQKrmdyw1es7Pm

    # image_noise_0_2_vae_state_0 https://drive.google.com/open?id=1xk82nhS2QwKPmU3VTeQjEHQ6BaywsTZm
    # encoding_noise_0_2_vae_state_0 https://drive.google.com/open?id=14FoCDyyKx3uFmaceoZ5j3CbnsK5vrpbH
    # control_vae_state_0 https://drive.google.com/open?id=1-BsE3nBnaG5GWDef0Ou9f4wrjiY9Aa43
    # control_vae_state_1 https://drive.google.com/open?id=107IyNz4bOj_r3W94AKRYgK8HuuuVKvc0

    ids = ['1IMRxCoRjgnc7bqyu16qtMYFRgVDfsTaL', '13lOdLsLqpgRXpIY6-XorkDhQi8xT5u_I',
           '1jZi37FXyUd1zq0AC8ERTDg7YFocNkPrX', '1ohdXmA-7SKNjzVa8SRWQKrmdyw1es7Pm',
           '1xk82nhS2QwKPmU3VTeQjEHQ6BaywsTZm', '14FoCDyyKx3uFmaceoZ5j3CbnsK5vrpbH',
           '1-BsE3nBnaG5GWDef0Ou9f4wrjiY9Aa43', '107IyNz4bOj_r3W94AKRYgK8HuuuVKvc0']
    dests = ['resnet_local.pt', 'resnet_poisson.pt', 'resnet_gaussian.pt', 'resnet_blots.pt',
             'image_noise_0_2_vae_state_0.pt', 'encoding_noise_0_2_vae_state_0.pt',
             'control_vae_state_0.pt', 'control_vae_state_1.pt']


    for i, j in zip(ids, dests):
        download_file_from_google_drive(i, j)

    num_epochs = 2
    batch_size = 200
    learning_rate = 0.001

    train_check = False
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    evens = list(range(0, len(trainset), 2))
    evens = random.sample(evens, 100)

    trainset_1 = torch.utils.data.Subset(trainset, evens)

    trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    evens_test = list(range(0, len(testset), 2))
    evens_test = random.sample(evens_test, 100)

    testset_1 = torch.utils.data.Subset(testset, evens_test)
    testloader = torch.utils.data.DataLoader(testset_1, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Samples for preliminary visualizations
    trainloader_play = torch.utils.data.DataLoader(trainset, batch_size=5,
                                                   shuffle=True, num_workers=1)

    # functions to show an image
    def imshow2(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader_play)
    images, labels = dataiter.next()

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    imshow2(torchvision.utils.make_grid(images))

    # Add some random noise to sampled images
    gaussian_noisy_image = random_noise(images, mode="gaussian")
    poisson_noisy_image = random_noise(images, mode="poisson")
    salt_noisy_image = random_noise(images, mode="salt")
    pepper_noisy_image = random_noise(images, mode="pepper")
    sp_noisy_image = random_noise(images, mode="s&p")
    speckle_noisy_image = random_noise(images, mode="speckle")
    localvar_noisy_image = random_noise(images, mode="localvar")

    imshow2(torchvision.utils.make_grid(torch.Tensor(list(gaussian_noisy_image))))
    imshow2(torchvision.utils.make_grid(torch.Tensor(list(poisson_noisy_image))))
    imshow2(torchvision.utils.make_grid(torch.Tensor(list(salt_noisy_image))))
    imshow2(torchvision.utils.make_grid(torch.Tensor(list(pepper_noisy_image))))
    imshow2(torchvision.utils.make_grid(torch.Tensor(list(sp_noisy_image))))
    imshow2(torchvision.utils.make_grid(torch.Tensor(list(speckle_noisy_image))))
    imshow2(torchvision.utils.make_grid(torch.Tensor(list(localvar_noisy_image))))

    number = 0
    modes = ['gaussian', 'poisson', 'localvar', 's&p']
    for i in range(len(modes)):
        print("CNN Noise type: " + modes[i])
        run_cuda_model(modes[i], number, trainloader, testloader, num_epochs, train_check)
        print('--------------------------------------')
        print('--------------------------------------')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    control_model = VAE(batch_size)
    if torch.cuda.is_available():
        control_model = control_model.cuda()
    optimizer = torch.optim.Adam(control_model.parameters(), lr=learning_rate)

    plot_num = 2

    if train_check:
        train_losses, test_losses = train_vae(control_model, optimizer,
                                              train_loader, test_loader,
                                              batch_size, num_epochs)
        torch.save(control_model.state_dict(), 'control_vae_state_2.pt')
        graph_loss(train_losses, test_losses, "vae_loss"+str(plot_num), num_epochs)
    graph_norms(control_model, testset, test_loader, batch_size)
    vae_compare_samples(control_model, test_loader, batch_size)

    image_noise_model = VAE(batch_size)
    if torch.cuda.is_available():
        image_noise_model = image_noise_model.cuda()
    optimizer = torch.optim.Adam(image_noise_model.parameters(), lr=learning_rate)

    if train_check:
        train_losses, test_losses = train_vae(image_noise_model, optimizer,
                                              train_loader, test_loader,
                                              batch_size, num_epochs,
                                              add_noise=True, noise_size=0.2)
        torch.save(image_noise_model.state_dict(), 'image_noise_0_2_vae_state_2.pt')
        graph_loss(train_losses, test_losses, "image_noise_loss"+str(plot_num), num_epochs)
    graph_norms(image_noise_model, testset, test_loader, batch_size)
    vae_compare_samples(image_noise_model, test_loader, batch_size)

    encoding_noise_model = VAE(batch_size, noise_size=0.2)
    if torch.cuda.is_available():
        encoding_noise_model = encoding_noise_model.cuda()
    optimizer = torch.optim.Adam(encoding_noise_model.parameters(), lr=learning_rate)

    if train_check:
        train_losses, test_losses = train_vae(encoding_noise_model,
                                              optimizer, train_loader,
                                              test_loader, batch_size,
                                              num_epochs)
        torch.save(encoding_noise_model.state_dict(), 'encoding_noise_0_2_vae_state_2.pt')
        graph_loss(train_losses, test_losses, "encoding_noise_loss"+str(plot_num), num_epochs)
    graph_norms(encoding_noise_model, testset, test_loader, batch_size)
    vae_compare_samples(encoding_noise_model, test_loader, batch_size)

    print("Control Noise Model")
    control_model = VAE(batch_size)
    if torch.cuda.is_available():
        control_model = control_model.cuda()
    if torch.cuda.is_available():
        control_model.load_state_dict(torch.load('control_vae_state_0.pt'))
    else:
        control_model.load_state_dict(torch.load('control_vae_state_0.pt', map_location=torch.device('cpu')))
    vae_compare_samples(control_model, test_loader, batch_size)
    if torch.cuda.is_available():
        control_model.load_state_dict(torch.load('control_vae_state_1.pt'))
    else:
        control_model.load_state_dict(torch.load('control_vae_state_1.pt', map_location=torch.device('cpu')))
    vae_compare_samples(control_model, test_loader, batch_size)

    print("Image Noise Model")
    image_noise_model = VAE(batch_size)
    if torch.cuda.is_available():
        image_noise_model = image_noise_model.cuda()
    if torch.cuda.is_available():
        image_noise_model.load_state_dict(torch.load('image_noise_0_2_vae_state_0.pt'))
    else:
        image_noise_model.load_state_dict(torch.load('image_noise_0_2_vae_state_0.pt', map_location=torch.device('cpu')))
    vae_compare_samples(image_noise_model, test_loader, batch_size)

    print("Encoding Noise Model")
    encoding_noise_model = VAE(batch_size, noise_size=0.2)
    if torch.cuda.is_available():
        encoding_noise_model = encoding_noise_model.cuda()
    if torch.cuda.is_available():
        encoding_noise_model.load_state_dict(torch.load('encoding_noise_0_2_vae_state_0.pt'))
    else:
        encoding_noise_model.load_state_dict(
            torch.load('encoding_noise_0_2_vae_state_0.pt', map_location=torch.device('cpu')))
    vae_compare_samples(encoding_noise_model, test_loader, batch_size)
