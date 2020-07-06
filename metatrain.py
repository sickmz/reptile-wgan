import torch
from args import get_args
from torch import nn
from torch.autograd import Variable

# Parsing params
args = get_args()

INPUT_SIZE = 784
REAL_LABEL = 1
FAKE_LABEL = 0

# Loss
criterion = nn.MSELoss() 

dom_weight = 1
adv_weight = 1

# Compute gradient penalty: (L2_norm(dy/dx) - 1)**2
def gradient_penalty(y, x):
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True
    )[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm - 1)**2)


def train_meta_learner( model_d, model_g, cloned_d, cloned_g, meta_optimizer_d, meta_optimizer_g, full_loader, loader, metal_iteration, metal_epochs, d=True):

    # Main loop

    # Update learning rate
    meta_lr = args.meta_lr #args.meta_lr * (1. - metal_iteration/float(metal_epochs))
    set_learning_rate(meta_optimizer_d, meta_lr)
    set_learning_rate(meta_optimizer_g, meta_lr)

    # Clone models
    net_d = model_d.clone()
    optimizer_d = get_optimizer(net_d)

    net_g = model_g.clone()
    optimizer_g = get_optimizer(net_g)

    # Sample base task from Meta-Train
    full_train_iter = make_infinite(full_loader)
    train_iter = make_infinite(loader)

    # Update fast net 
    ret_values_d, ret_values_g = do_learning(
        net_d, net_g, optimizer_d, optimizer_g, full_train_iter, train_iter,
        args.iterations, d=d)

    model_d.point_grad_to(net_d)
    model_g.point_grad_to(net_g)

    meta_optimizer_d.step()
    meta_optimizer_g.step()

    return ret_values_d, ret_values_g


def Discriminator_training( batch_size, optimizer, model_d, model_g, full_x, train_x, train_y, noise, label):
    
    train_x = train_x.cuda()
    full_x = full_x.cuda()

    label_real = Variable(label).cuda()
    # Discriminator training, real examples
    output, dom_out = model_d(train_x)
    optimizer.zero_grad()
    errD_real = -output.mean() 

    label_real.resize_(dom_out.data.size()).fill_(REAL_LABEL).cuda()
    err_dom_real = criterion(dom_out, label_real)
    realD_mean = output.data.cpu().mean()
    noise.resize_(batch_size, 100, 1, 1).normal_(0, 1)
    [noisev] = Variable_([noise])

    # Fake examples
    g_out = model_g(noisev)
    output, _ = model_d(g_out.detach())
    errD_fake =  output.mean()
    fakeD_mean = output.data.cpu().mean()

    # Compute loss for gradient penalty
    alpha = torch.rand(train_x.size(0), 1, 1, 1).cuda()
    x_hat = (alpha * train_x.data + (1 - alpha) * g_out.data).requires_grad_(True)
    out_src, _ = model_d(x_hat)
    d_loss_gp = gradient_penalty(out_src, x_hat)

    # Domain
    _, out_dom_fake = model_d(full_x)

    label_fake = torch.FloatTensor(args.batch_size).cuda().fill_(FAKE_LABEL)
    label_fake.resize_(out_dom_fake.data.size())

    err_dom_fake = criterion(out_dom_fake, label_fake)
    loss =  errD_real + 10 * err_dom_real + errD_fake + err_dom_fake + 10*d_loss_gp
    loss.backward()

    # Next
    optimizer.step()

    return errD_real, errD_fake, fakeD_mean, realD_mean


def Generator_training(batch_size, optimizer, model_d, model_g, noise, label):

    noise.resize_(batch_size, 100, 1, 1).normal_(0, 1)
    [noisev] = Variable_([noise])
    label_real = Variable(label).cuda()

    # GAN training
    g_out = model_g(noisev)
    output, out_dom_real = model_d(g_out)
    err = - output.mean()
    label_real.resize_(out_dom_real.data.size()).fill_(REAL_LABEL).cuda()
    
    err_dom_real = criterion(out_dom_real, label_real)
    g_err = adv_weight * err + dom_weight * err_dom_real
    optimizer.zero_grad()
    g_err.backward()

    # Next
    optimizer.step()

    return err


def do_learning(model_d, model_g, optimizer_d, optimizer_g, full_train_iter, train_iter,iterations, d=True):

    it = 0
    while(it < iterations):   
        
        noise = torch.FloatTensor(args.batch_size, 100, 1, 1).cuda()
        label = torch.FloatTensor(args.batch_size).cuda()

        train_x,train_y = next(iter(train_iter))
        full_x,_ = next(iter(full_train_iter)) 
        
        for i in range(5):
            noise = torch.FloatTensor(args.batch_size, 100, 1, 1).cuda()
            label = torch.FloatTensor(args.batch_size).cuda()

            train_x,train_y = next(iter(train_iter))
            full_x,_ = next(iter(full_train_iter)) 

            # Stop condition
            actual_batch_size = train_x.size(0)
            if actual_batch_size % args.batch_size != 0:
                continue
            if train_x.size(0) != full_x.size(0):
                continue

            ret_values_d = Discriminator_training(
                    actual_batch_size,
                    optimizer_d, model_d, model_g,
                    full_x,
                    train_x, train_y,                
                    noise, label
            )

        noise = torch.FloatTensor(args.batch_size, 100, 1, 1).cuda()
        label = torch.FloatTensor(args.batch_size).cuda()

        train_x,train_y = next(iter(train_iter))
        full_x,_ = next(iter(full_train_iter)) 
        
        # Stop condition
        actual_batch_size = train_x.size(0)
        if actual_batch_size % args.batch_size != 0:
            continue
        if train_x.size(0) != full_x.size(0):
            continue
        # Run Generator training
        ret_values_g = Generator_training(
                actual_batch_size,
                optimizer_g, model_d, model_g,
                noise, label
        )

        it += 1

    return ret_values_d, ret_values_g


def do_evaluation(model_g, fixed_noise):

    g_out = model_g(fixed_noise)
    return g_out.cpu()


def test_meta_learner(model_g, model_d, full_loader, loader, fixed_noise):

    full_test_iter = make_infinite(full_loader)
    test_iter = make_infinite(loader)

    net_d = model_d.clone()
    optimizer_d = get_optimizer(net_d)

    net_g = model_g.clone()
    optimizer_g = get_optimizer(net_g)

    do_learning(net_d, net_g, optimizer_d, optimizer_g, full_test_iter, test_iter,
                args.test_iterations, d=True)

    return do_evaluation(net_g, fixed_noise)


# Utils
def get_optimizer(net, state=None):

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.9))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def initialize_meta_optimizer(model):

    return torch.optim.SGD(model.parameters(), lr=args.meta_lr, momentum=0.5)


def make_infinite(dataloader):

    while True:
        for x in dataloader:
            yield x

# Make variable cuda depending on the arguments
def Variable_(tensor, *args_, **kwargs):

    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs)
                for key, v in tensor.items()}
    # Normal tensor
    return Variable(tensor, *args_, **kwargs).cuda()
