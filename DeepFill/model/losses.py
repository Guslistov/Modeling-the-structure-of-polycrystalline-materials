import torch

def gradient_penalty(discriminator, real, fake, lambda_gp=1):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolates = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return penalty


def ls_loss_d(pos, neg, value=1.):
    """
    least-square loss discriminator
    """
    l2_pos = torch.mean((pos-value)**2)
    l2_neg = torch.mean(neg**2)
    d_loss = 0.5*l2_pos + 0.5*l2_neg 
    return d_loss

def ls_loss_g(neg, value=1.):    
    """
    least-square loss generator
    """
    g_loss = torch.mean((neg-value)**2)
    return g_loss

def hinge_loss_d2(pos, neg):
    """
    hinge loss discriminator
    """
    hinge_pos = torch.mean(torch.relu(1-pos))
    hinge_neg = torch.mean(torch.relu(1+neg))
    d_loss = 0.5*hinge_pos + 0.5*hinge_neg   
    return d_loss

def hinge_loss_d(pos, neg):
    pos = pos.view(-1)
    neg = neg.view(-1)
    #print("d_real shape:", pos.shape, "d_gen shape:", neg.shape)
    #print(f"d_real: {pos.mean().item():.3f}, d_gen: {neg.mean().item():.3f}")
    hinge_pos = torch.mean(torch.relu(1 - pos))
    hinge_neg = torch.mean(torch.relu(1 + neg))
    d_loss = 0.5 * hinge_pos + 0.5 * hinge_neg
    return d_loss

def hinge_loss_g2(neg):
    """
    hinge loss generator
    """
    g_loss = -torch.mean(neg)
    return g_loss

def hinge_loss_g(neg):
    neg = neg.view(-1)
    g_loss = -torch.mean(neg)
    return g_loss

def w_loss_d(pos, neg):
    """
    Wasserstein loss for discriminator (critic)
    pos: scores for real samples (D(x))
    neg: scores for fake samples (D(G(z)))
    Maximizes: D(x) - D(G(z))
    """
    pos = pos.view(-1)
    neg = neg.view(-1)
    d_loss = - (torch.mean(pos) - torch.mean(neg))
    return d_loss

def w_loss_g(neg):
    """
    Wasserstein loss for generator
    neg: scores for fake samples (D(G(z)))
    Maximizes: D(G(z))
    """
    neg = neg.view(-1)
    g_loss = -torch.mean(neg)
    return g_loss