import torch
import numpy as np
import os
import grid_sampler
from cosine_sampler_2d import CosineSampler2d


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    
    np.random.seed(51) 
    torch.manual_seed(51)
    torch.cuda.manual_seed_all(51)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(51)
    torch.backends.cudnn.deterministic = True
    
    off= True
    numb = 100000
    n_cell = 96
    cell_dim = 4
    step = 'cosine'
    padding_mode = 'zeros' # border, reflection
    align_corners = True
    cells = torch.nn.Parameter(torch.rand([n_cell, cell_dim, 16, 16], device = 'cuda')).requires_grad_(True)
    
    yx = np.random.rand(numb, 2)
    yx[..., 1] = yx[..., 1]*2-1
    yx_f = torch.tensor(yx, requires_grad=True).float()
    y = yx_f[:, 0:1]
    x = yx_f[:, 1:2]
    x = x.to('cuda')
    y = y.to('cuda')

    grid = torch.cat([y*2-1, x], -1)

    grid = grid.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])
    
    val2 = CosineSampler2d.apply(cells, grid, padding_mode, align_corners, 'cosine', True)

    net= []
    net.append(torch.nn.Linear(cell_dim, 16))
    net.append(torch.nn.Tanh())
    net.append(torch.nn.Linear(16, 1))
    # deploy layers
    net = torch.nn.Sequential(*net)
    
    val2 = val2.to("cpu")
    
    val2 = val2.sum(0).view(cell_dim,-1).t()
    val2 = net(val2)               

    print("---u2cell---")
    u2_cell = torch.autograd.grad(
        val2, cells, 
        grad_outputs=torch.ones_like(val2),
        retain_graph=True,
        create_graph=True
    )[0]

    print("---u2x---")
    u2_x = torch.autograd.grad(
        val2, x, 
        grad_outputs=torch.ones_like(val2),
        retain_graph=True,
        create_graph=True
    )[0]


    print("---u2y==--")
    u2_y = torch.autograd.grad(
        val2, y, 
        grad_outputs=torch.ones_like(val2),
        retain_graph=True,
        create_graph=True
    )[0]


    print("---u2xx==--")
    u2_xx = torch.autograd.grad(
        u2_x, x, 
        grad_outputs=torch.ones_like(u2_x),
        retain_graph=True,
        create_graph=True
    )[0]

    
    print("--u2_yy--")
    u2_yy = torch.autograd.grad(
        u2_y, y, 
        grad_outputs=torch.ones_like(u2_y),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u2_x_cell--")
    u2_x_cell = torch.autograd.grad(
        u2_x, cells, 
        grad_outputs=torch.ones_like(u2_x),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u2_y_cell--")
    u2_y_cell = torch.autograd.grad(
        u2_y, cells, 
        grad_outputs=torch.ones_like(u2_y),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u2_xx_cell--")
    u2_xx_cell = torch.autograd.grad(
        u2_xx, cells, 
        grad_outputs=torch.ones_like(u2_xx),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u2_yy_cell--")
    u2_yy_cell = torch.autograd.grad(
        u2_yy, cells, 
        grad_outputs=torch.ones_like(u2_yy),
        retain_graph=True,
        create_graph=True
    )[0]

    
    val = grid_sampler.grid_sample_2d(cells, grid, step='cosine', offset=True) 
    val = val.to("cpu")
    val = val.sum(0).view(cell_dim,-1).t()
    val = net(val)
    
    
    print("---ucell---")
    u_cell = torch.autograd.grad(
        val, cells, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]

    print("---ux---")
    u_x = torch.autograd.grad(
        val, x, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]

    print("---uy==--")
    u_y = torch.autograd.grad(
        val, y, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]

    print("---uxx==--")
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    
    print("--u_yy--")
    u_yy = torch.autograd.grad(
        u_y, y, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u_x_cell--")
    u_x_cell = torch.autograd.grad(
        u_x, cells, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u_y_cell--")
    u_y_cell = torch.autograd.grad(
        u_y, cells, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u_xx_cell--")
    u_xx_cell = torch.autograd.grad(
        u_xx, cells, 
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u_yy_cell--")
    u_yy_cell = torch.autograd.grad(
        u_yy, cells, 
        grad_outputs=torch.ones_like(u_yy),
        retain_graph=True,
        create_graph=True
    )[0]

    number = numb * n_cell * cell_dim

    print('val == val2, max_error: {} at {}'.format((val.reshape(-1)-val2.reshape(-1)).abs().max(),(val.reshape(-1)-val2.reshape(-1)).abs().argmax()))
    print('u_cell == u2_cell, max_error: {} at {}'.format((u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().max(),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().argmax()))
    print('u_x == u2_x, max_error: {} at {}'.format((u_x.reshape(-1)-u2_x.reshape(-1)).abs().max(),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().argmax()))
    print('u_y == u2_y, max_error: {} at {}'.format((u_y.reshape(-1)-u2_y.reshape(-1)).abs().max(),(u_y.reshape(-1)-u2_y.reshape(-1)).abs().argmax()))
    print('u_xx == u2_xx, max_error: {} at {}'.format((u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().max(),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().argmax()))
    print('u_yy == u2_yy, may_error: {} at {}'.format((u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().max(),(u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().argmax()))
    print('u_x_cell == u2_x_cell, max_error: {} at {}'.format((u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().max(),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().argmax()))
    print('u_y_cell == u2_y_cell, max_error: {} at {}'.format((u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().max(),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().argmax()))
    print('u_xx_cell == u2_xx_cell, max_error: {} at {}'.format((u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().max(),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().argmax()))
    print('u_yy_cell == u2_yy_cell, max_error: {} at {}'.format((u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().max(),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().argmax()))
    
    f2_pred = u2_y*2 + 5*(val2.to('cuda')**3) - 5*val2.to('cuda') - 0.0001*u2_xx  # u2_yy + u2_xx + val2.to('cuda') #u2_y + val2.to('cuda')*u2_x -(0.01/np.pi)*u2_xx #
    loss2_f = torch.mean((f2_pred)**2)
    
    print("----dloss2----")
    dloss2 = torch.autograd.grad(
        loss2_f, cells, 
        grad_outputs=torch.ones_like(loss2_f),
        retain_graph=True,
        create_graph=True
    )[0]
    
    f_pred = u_y*2 + 5*(val.to('cuda')**3) - 5*val.to('cuda') - 0.0001*u_xx  #u_yy + u_xx + val.to('cuda') #u_y + val.to('cuda')*u_x -(0.01/np.pi)*u_xx #
    loss_f = torch.mean((f_pred)**2)
    
    dloss = torch.autograd.grad(
        loss_f, cells, 
        grad_outputs=torch.ones_like(loss_f),
        retain_graph=True,
        create_graph=True
    )[0]

    
    print('dloss == dloss2: {}, max_error: {} at {}'.format(((dloss.reshape(-1)-dloss2.reshape(-1)).abs()<1e-4).sum()==(number),(dloss.reshape(-1)-dloss2.reshape(-1)).abs().max(),(dloss.reshape(-1)-dloss2.reshape(-1)).abs().argmax()))
    print('dloss error: ',np.testing.assert_allclose((dloss).reshape(-1).detach().cpu().numpy(), (dloss2).reshape(-1).detach().cpu().numpy(), rtol=1e-4, atol=0))
