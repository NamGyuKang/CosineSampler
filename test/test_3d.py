import torch
import numpy as np
import os
import grid_sampler
from cosine_sampler_2d import CosineSampler3d


    
if __name__ == "__main__":
    torch.manual_seed(6)
    torch.cuda.manual_seed(6)
    padding_mode = 'zeros'
    align_corners = True
    point_num = 100000
    n_cell = 50
    cell_dim = 4
    step = True

    cells = torch.nn.Parameter(torch.rand([n_cell, cell_dim, 16, 16, 16], device = 'cuda')).requires_grad_(True)

    yxz = np.random.rand(point_num, 3)
    yxz_f = torch.tensor(yxz, requires_grad=True).float()
    y = yxz_f[:, 0:1]
    x = yxz_f[:, 1:2]
    z = yxz_f[:, 2:3]

    x = x.to("cuda")
    y = y.to("cuda")
    z = z.to("cuda")

    grid = torch.cat([z, y, x], -1)
    grid = grid.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1, 1])

    val2 = CosineSampler3d.apply(cells, grid, padding_mode, align_corners, 'cosine', False)
    
    net= []
    net.append(torch.nn.Linear(4, 16))
    net.append(torch.nn.Tanh())
    for i in range(2-2): 
        net.append(torch.nn.Linear(16, 16))
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

    print("---u2z==--")
    u2_z = torch.autograd.grad(
        val2, z, 
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

    print("--u2_zz--")
    u2_zz = torch.autograd.grad(
        u2_z, z, 
        grad_outputs=torch.ones_like(u2_z),
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

    print("--u2_z_cell--")
    u2_z_cell = torch.autograd.grad(
        u2_z, cells, 
        grad_outputs=torch.ones_like(u2_z),
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

    print("--u2_zz_cell--")
    u2_zz_cell = torch.autograd.grad(
        u2_zz, cells, 
        grad_outputs=torch.ones_like(u2_zz),
        retain_graph=True,
        create_graph=True
    )[0]


    val = grid_sampler.grid_sample_3d(cells, grid, step='cosine', offset=False).unsqueeze(2)
    val = val.to("cpu")
    val = val.sum(0).view(cell_dim,-1).t()
    val = net(val)

    u_cell = torch.autograd.grad(
        val, cells, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_x = torch.autograd.grad(
        val, x, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_y = torch.autograd.grad(
        val, y, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_yy = torch.autograd.grad(
        u_y, y, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_z = torch.autograd.grad(
        val, z, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_zz = torch.autograd.grad(
        u_z, z, 
        grad_outputs=torch.ones_like(u_z),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_x_cell = torch.autograd.grad(
        u_x, cells, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_y_cell = torch.autograd.grad(
        u_y, cells, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_z_cell = torch.autograd.grad(
        u_z, cells, 
        grad_outputs=torch.ones_like(u_z),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_xx_cell = torch.autograd.grad(
        u_xx, cells, 
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_yy_cell = torch.autograd.grad(
        u_yy, cells, 
        grad_outputs=torch.ones_like(u_yy),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_zz_cell = torch.autograd.grad(
        u_zz, cells, 
        grad_outputs=torch.ones_like(u_zz),
        retain_graph=True,
        create_graph=True
    )[0]
    
    print('val == val2, max_error: {} at {}'.format((val.reshape(-1)-val2.reshape(-1)).abs().max(),(val.reshape(-1)-val2.reshape(-1)).abs().argmax()))
    print('u_cell == u2_cell, max_error: {} at {}'.format((u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().max(),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().argmax()))
    print('u_x == u2_x, max_error: {} at {}'.format((u_x.reshape(-1)-u2_x.reshape(-1)).abs().max(),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().argmax()))
    print('u_y == u2_y, max_error: {} at {}'.format((u_y.reshape(-1)-u2_y.reshape(-1)).abs().max(),(u_y.reshape(-1)-u2_y.reshape(-1)).abs().argmax()))
    print('u_z == u2_z, max_error: {} at {}'.format((u_z.reshape(-1)-u2_z.reshape(-1)).abs().max(),(u_z.reshape(-1)-u2_z.reshape(-1)).abs().argmax()))
    print('u_xx == u2_xx, max_error: {} at {}'.format((u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().max(),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().argmax()))
    print('u_yy == u2_yy, may_error: {} at {}'.format((u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().max(),(u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().argmax()))
    print('u_zz == u2_zz, may_error: {} at {}'.format((u_zz.reshape(-1)-u2_zz.reshape(-1)).abs().max(),(u_zz.reshape(-1)-u2_zz.reshape(-1)).abs().argmax()))
    print('u_x_cell == u2_x_cell, max_error: {} at {}'.format((u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().max(),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().argmax()))
    print('u_y_cell == u2_y_cell, max_error: {} at {}'.format((u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().max(),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().argmax()))
    print('u_z_cell == u2_z_cell, max_error: {} at {}'.format((u_z_cell.reshape(-1)-u2_z_cell.reshape(-1)).abs().max(),(u_z_cell.reshape(-1)-u2_z_cell.reshape(-1)).abs().argmax()))
    print('u_xx_cell == u2_xx_cell, max_error: {} at {}'.format((u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().max(),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().argmax()))
    print('u_yy_cell == u2_yy_cell, max_error: {} at {}'.format((u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().max(),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().argmax()))
    print('u_zz_cell == u2_zz_cell, max_error: {} at {}'.format((u_zz_cell.reshape(-1)-u2_zz_cell.reshape(-1)).abs().max(),(u_zz_cell.reshape(-1)-u2_zz_cell.reshape(-1)).abs().argmax()))
     
    f2_pred = u2_xx + u2_yy + u2_zz + val2.to('cuda') #u2_y*2 + 5*(val2.to('cuda')**3) - 5*val2.to('cuda') - 0.0001*u2_xx  # u2_yy + u2_xx + val2.to('cuda') #u2_y + val2.to('cuda')*u2_x -(0.01/np.pi)*u2_xx #
    loss2_f = torch.mean((f2_pred)**2)
    
    print("----dloss2----")
    dloss2 = torch.autograd.grad(
        loss2_f, cells, 
        grad_outputs=torch.ones_like(loss2_f),
        retain_graph=True,
        create_graph=True
    )[0]
    
    f_pred = u_xx + u_yy + u_zz + val.to('cuda') #u2_y*2 + 5*(val2.to('cuda')**3) - 5*val2.to('cuda') - 0.0001*u2_xx  # u2_yy + u2_xx + val2.to('cuda') #u2_y + val2.to('cuda')*u2_x -(0.01/np.pi)*u2_xx #
    loss_f = torch.mean((f_pred)**2)
    
    dloss = torch.autograd.grad(
        loss_f, cells, 
        grad_outputs=torch.ones_like(loss_f),
        retain_graph=True,
        create_graph=True
    )[0]

    
    print('dloss == dloss2, max_error: {} at {}'.format((dloss.reshape(-1)-dloss2.reshape(-1)).abs().max(),(dloss.reshape(-1)-dloss2.reshape(-1)).abs().argmax()))
    print('dloss error: ',np.testing.assert_allclose((dloss).reshape(-1).detach().cpu().numpy(), (dloss2).reshape(-1).detach().cpu().numpy(), rtol=1e-4, atol=0))
    
    exit(1)