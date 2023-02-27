import torchvision
import torchvision.transforms as transforms
import torch

root_path = '../'
def _baseset_picker(args):
    size = 32
    if args.baseset == 'CIFAR10':
        ''' best transforms - figure out later (LF 06/11/21)
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        clean_trainset = torchvision.datasets.CIFAR10(
            root=f'{root_path}/data', train=True, download=True, transform=transform_train)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        
        testset = torchvision.datasets.CIFAR10(
            root=f'{root_path}/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    elif args.baseset == 'FashionMNIST':
        transform=transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        clean_trainset = torchvision.datasets.FashionMNIST(
            root=f'{root_path}/data', train=True, download=True, transform=transform)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        
        testset = torchvision.datasets.FashionMNIST(
            root=f'{root_path}/data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    elif args.baseset == 'MNIST':
        transform=transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        clean_trainset = torchvision.datasets.MNIST(
            root=f'{root_path}/data', train=True, download=True, transform=transform)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        
        testset = torchvision.datasets.MNIST(
            root=f'{root_path}/data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    return clean_trainloader, testloader

def get_data(args):
    print('==> Preparing data..')
    clean_trainloader, testloader = _baseset_picker(args)

    return clean_trainloader, testloader

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''
    a = img2 - img1
    b = img3 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    return a, b_orthog, b, coords


class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
                    range_l=.1, range_r=.1):
        self.base_img = base_img
        self.vec1 = vec1
        self.vec2 = vec2
        self.coords = coords
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        grid = torch.meshgrid([list1,list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def make_planeloader(images, args):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])

    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=args.resolution, range_l=args.range_l, range_r=args.range_r)

    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=256, shuffle=False, num_workers=2)
    return planeloader