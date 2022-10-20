import torch
from torch.utils.cpp_extension import load
import grid_sample_temp
import os

import time


_cosine = load(name="_cosine", sources = ['/hdd/kng/CosineSampler/cosine_sampler_kernel.cu', '/hdd/kng/CosineSampler/cosine_sampler.cpp'])


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def padding_mode_enum(padding_mode):
    if padding_mode == 'zeros':
        return 0
    elif padding_mode == 'border':
        return 1
    else :
        return 2


class CosineSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode='zeros', align_corners = True):
        ''' offset merge kernel'''
        
        ctx.offset  =torch.linspace(0, 1-(1/input.shape[0]), input.shape[0]).to('cuda')
        ctx.padding_mode = padding_mode
        output = _cosine.forward(input, grid, ctx.offset, padding_mode_enum(padding_mode=padding_mode), align_corners)
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners

        return output

    @staticmethod
    def backward(ctx, gradOut):
        
        input, grid = ctx.saved_tensors
        
        d_input, d_grid = CosineSamplerBackward.apply(input, grid, gradOut.contiguous(), ctx.offset, ctx.padding_mode, ctx.align_corners)

        return d_input, d_grid, None, None, None


class CosineSamplerBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, gOut, offset, padding_mode="zeros", align_corners=True):
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        ctx.offset = offset
        

        gInput, gGrid = _cosine.backward(gOut, input, grid, offset, padding_mode_enum(padding_mode),
                                            ctx.align_corners, input.requires_grad) 
        
        ctx.save_for_backward(input, grid, gOut)

        return gInput, gGrid

    @staticmethod
    def backward(ctx, gOutInput, gOutGrid):
        input, grid, gOut = ctx.saved_tensors

        gInput, gGrid, ggOut = CosineSamplerBackwardBackward.apply(input, grid, gOut, gOutInput.contiguous(), gOutGrid.contiguous(), ctx.offset, ctx.padding_mode, ctx.align_corners)


        return gInput, gGrid, ggOut, None, None, None, None

class CosineSamplerBackwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, gOut, gOutInput, gOutGrid, offset, padding_mode="zeros", align_corners=True):
        
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        ctx.offset = offset

        
        input_requires_grad = gOutInput is not None and (gOutInput != 0.).any().item()

        gInput, gGrid, ggOut = _cosine.backward_backward(gOutInput, gOutGrid, input, grid, gOut, offset,
                                                                    padding_mode_enum(padding_mode), align_corners, input_requires_grad)
        ctx.save_for_backward(input, grid, gOut, gOutGrid)
        
        return gInput, gGrid, ggOut

    @staticmethod
    def backward(ctx, gOutgInput, gOutgGrid, gOutggOut):
        align_corners = ctx.align_corners
        padding_mode = ctx.padding_mode
        input, grid, gOut, gOutGrid,  = ctx.saved_tensors 
        

        input_requires_grad = gOutgInput is not None and (gOutgInput != 0.).any().item()
        
        gInput, ggOut = _cosine.backward_backward_backward(input, grid, gOut, gOutggOut.contiguous(), gOutGrid, gOutgGrid.contiguous(), ctx.offset,
                                                                    padding_mode_enum(padding_mode), align_corners, input_requires_grad) 

        b_input, _, _ = CosineSamplerBackwardBackward.apply(input, grid, gOutggOut.contiguous(), torch.ones_like(gOutgInput), gOutGrid, ctx.offset)

        return gInput+b_input, None, ggOut, None, None, None, None, None, None


# def source_term( y, x):
#     u_gt = exact_u(y, x)
#     u_yy = -(4*torch.pi)**2 * u_gt
#     u_xx = -(1*torch.pi)**2 * u_gt
#     return  u_yy + u_xx + 1*u_gt
# def exact_u( y, x):
#         return torch.sin(4*torch.pi*y) * torch.sin(1*torch.pi*x)

# def generate_train_data( num_train):
#     # colocation points
#     yc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
#     xc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
#     with torch.no_grad():
#         uc = source_term(yc, xc)
#     # requires grad
#     yc.requires_grad = True
#     xc.requires_grad = True
#     # boundary points
#     north = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
#     west = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
#     south = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
#     east = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
#     yb = torch.cat([
#         torch.ones((num_train, 1)), west,
#         torch.ones((num_train, 1)) * -1, east
#         ])
#     xb = torch.cat([
#         north, torch.ones((num_train, 1)) * -1,
#         south, torch.ones((num_train, 1))
#         ])
#     ub = exact_u(yb, xb)
#     return yc, xc, uc, yb, xb, ub
        

if __name__ == '__main__':
    import numpy as np
    
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
    import numpy as np
    cells = torch.nn.Parameter(torch.rand([n_cell, cell_dim, 16, 16], device = 'cuda')).requires_grad_(True)
    # cells = cells.data.uniform_(-1e-05, 1e-05).requires_grad_(True)
    # y, x, uc, yb, xb, ub = generate_train_data(numb)
    # tx = np.random.rand(numb, 2)
    # tx[..., 0] = tx[..., 0] # t -> [t_start, t_start + t_range]
    # tx[..., 1] =  2*np.pi*tx[..., 1] # x -> [0, 2*pi], t -> [0, 1]
    # tx_f = torch.tensor(tx, requires_grad=True).float()
    # y = tx_f[:,0:1]
    # x = tx_f[:,1:2]

    # tx = np.random.rand(numb, 2)
    # # tx[..., 0] = tx[..., 0] # t -> [   tx[..., 1] = 2 * tx[..., 1] - 1 # x -> [-1, 1], t -> [0, 1]
    # tx_f = torch.tensor(tx, requires_grad=True).float()
    # y = tx_f[:,0:1]
    # x = tx_f[:,1:2]

    # # create IC
    # tx_ic = 2 * np.random.rand(numb, 2) - 1      # x_ic = -1 ~ +1
    # tx_ic[..., 0] = 0                                   # t_ic =  0

    # # create BC
    # tx_bc = np.random.rand(numb, 2)              # t_bc =  0 ~ +1
    # # tx_bc[..., 0] = tx_bc[..., 0]   # t -> [t_start, t_start + t_range]
    # tx_bc[..., 1] = 2 * np.round(tx_bc[..., 1]) - 1     # x_bc = -1 or +1
    # tx_ic_bc = torch.tensor(np.concatenate((tx_ic, tx_bc)), requires_grad=True).float()
    # y2 = tx_ic_bc[:, 0:1]
    # x2 = tx_ic_bc[:, 1:2]

    # # create output values for IC and BCs
    # u_ic = np.sin(-np.pi * tx_ic[..., 1, np.newaxis])        # u_ic = -sin(pi*x_ic)
    # u_bc = np.zeros((numb, 1))                        # u_bc = 0
    # u_train = torch.tensor(np.concatenate((u_ic, u_bc))).float()
    yx = np.random.rand(numb, 2)
    yx[..., 1] = yx[..., 1]*2-1
    yx_f = torch.tensor(yx, requires_grad=True).float()
    y = yx_f[:, 0:1]
    x = yx_f[:, 1:2]#:, :, 
    # y2 = yx_f[..., 0]
    # x2 = yx_f[..., 1]
    
    
    # # requires grad
    # y.requires_grad = True
    # x.requires_grad = True
    # x = x*2-1
    # y = y*2-1

    x = x.to('cuda')
    y = y.to('cuda')
    # uc = uc.to('cuda')

    grid = torch.cat([y*2-1, x], -1)
    # grid_ic = torch.cat([yb, xb], -1).to('cuda')

    # with torch.no_grad():
    grid = grid.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])
    # grid_ic = grid_ic.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])
    # grid = grid.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])

    # grid = torch.rand((n_cell, 1, numb, 2), dtype = torch.float32).requires_grad_(True).to('cuda')
    
    start = time.time()

    val2 = CosineSampler.apply(cells, grid, padding_mode, align_corners)

    # val2_ic = CosineSampler.apply(cells, grid_ic, padding_mode, align_corners, step)
    # val2_ic = val2_ic.to('cpu')
    end = time.time()

    print('custom forward: ',end - start)
    net= []
    net.append(torch.nn.Linear(cell_dim, 16))
    net.append(torch.nn.Tanh())
    # for i in range(2-2): 
    #     net.append(torch.nn.Linear(16, 16))
    #     net.append(torch.nn.Tanh())
    net.append(torch.nn.Linear(16, 1))


    # deploy layers
    net = torch.nn.Sequential(*net)
    
    val2 = val2.to("cpu")
    # val2 = torch.tanh(val2)
    # with torch.no_grad():
    
    val2 = val2.sum(0).view(cell_dim,-1).t()
    val2 = net(val2)        

    # val2_ic = val2_ic.sum(0).view(cell_dim,-1).t()
    # val2_ic = net(val2_ic)        
    

    print("---u2cell---")
    start = time.time()
    u2_cell = torch.autograd.grad(
        val2, cells, 
        grad_outputs=torch.ones_like(val2),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_cell time: ", end - start)

    print("---u2x---")
    start = time.time()
    u2_x = torch.autograd.grad(
        val2, x, 
        grad_outputs=torch.ones_like(val2),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()

    print("u_x time: ", end - start)


    print("---u2y==--")
    start = time.time()
    u2_y = torch.autograd.grad(
        val2, y, 
        grad_outputs=torch.ones_like(val2),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_y time: ", end - start)


    print("---u2xx==--")
    start = time.time()
    u2_xx = torch.autograd.grad(
        u2_x, x, 
        grad_outputs=torch.ones_like(u2_x),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_xx time: ", end - start)

    
    print("--u2_yy--")
    start = time.time()
    u2_yy = torch.autograd.grad(
        u2_y, y, 
        grad_outputs=torch.ones_like(u2_y),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_yy time: ", end - start)

    print("--u2_x_cell--")
    start = time.time()
    u2_x_cell = torch.autograd.grad(
        u2_x, cells, 
        grad_outputs=torch.ones_like(u2_x),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_x_cell time: ", end - start)
    print("shape check", u2_x_cell.shape)

    print("--u2_y_cell--")
    start = time.time()
    u2_y_cell = torch.autograd.grad(
        u2_y, cells, 
        grad_outputs=torch.ones_like(u2_y),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_y_cell time: ", end - start)
    print("shape check", u2_y_cell.shape)

    print("--u2_xx_cell--")
    start = time.time()
    u2_xx_cell = torch.autograd.grad(
        u2_xx, cells, 
        grad_outputs=torch.ones_like(u2_xx),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_xx_cell time: ", end - start)
    print("shape check", u2_xx_cell.shape)

    print("--u2_yy_cell--")
    start =time.time()
    u2_yy_cell = torch.autograd.grad(
        u2_yy, cells, 
        grad_outputs=torch.ones_like(u2_yy),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_yy_cell time: ", end - start)
    print("shape check", u2_yy_cell.shape)

    # print("--u2_xxx--")
    # u2_xxx = torch.autograd.grad(
    #     u2_xx, x, 
    #     grad_outputs=torch.ones_like(u2_xx),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]
    # print("shape check", u2_xxx.shape)

    # print("--u2_yyy--")
    # u2_yyy = torch.autograd.grad(
    #     u2_yy, y, 
    #     grad_outputs=torch.ones_like(u2_yy),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]
    # print("shape check", u2_yyy.shape)


    # make_dot(val2, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val", format="png")
    # make_dot(u2_x, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val_x", format="png")
    # make_dot(u2_xx, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val_xx", format="png")
    print("----")


    start = time.time()
    
    val = grid_sample_temp.grid_sample_2d(cells, grid, step=step, offset=off) 
    # val_ic = grid_sample_temp.grid_sample_2d(cells, grid_ic, step = step, offset = off)
    end = time.time()
    print('python forward: ', end - start)



    val = val.to("cpu")
    # val_ic = val_ic.to('cpu')

    # val = torch.tanh(val)
    # # val = torch.tanh(val)
    
    val = val.sum(0).view(cell_dim,-1).t()
    val = net(val)
    # val_ic = val_ic.sum(0).view(cell_dim,-1).t()
    # val_ic = net(val_ic)

    
    print("---u2cell---")
    start = time.time()
    u_cell = torch.autograd.grad(
        val, cells, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_cell time: ", end - start)

    print("---ux---")
    start = time.time()
    u_x = torch.autograd.grad(
        val, x, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()

    print("u_x time: ", end - start)


    print("---uy==--")
    start = time.time()
    u_y = torch.autograd.grad(
        val, y, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_y time: ", end - start)


    print("---uxx==--")
    start = time.time()
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_xx time: ", end - start)

    
    print("--u_yy--")
    start = time.time()
    u_yy = torch.autograd.grad(
        u_y, y, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_yy time: ", end - start)

    print("--u_x_cell--")
    start = time.time()
    u_x_cell = torch.autograd.grad(
        u_x, cells, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_x_cell time: ", end - start)
    print("shape check", u_x_cell.shape)

    print("--u_y_cell--")
    start = time.time()
    u_y_cell = torch.autograd.grad(
        u_y, cells, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_y_cell time: ", end - start)
    print("shape check", u_y_cell.shape)

    print("--u_xx_cell--")
    start = time.time()
    u_xx_cell = torch.autograd.grad(
        u_xx, cells, 
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_xx_cell time: ", end - start)
    print("shape check", u_xx_cell.shape)

    print("--u_yy_cell--")
    start =time.time()
    u_yy_cell = torch.autograd.grad(
        u_yy, cells, 
        grad_outputs=torch.ones_like(u_yy),
        retain_graph=True,
        create_graph=True
    )[0]
    end = time.time()
    print("u_yy_cell time: ", end - start)
    print("shape check", u_yy_cell.shape)

    import numpy as np    
    # print(np.testing.assert_allclose(u2_yy_cell.reshape(-1).detach().cpu().numpy(), u_yy_cell.reshape(-1).detach().cpu().numpy(), rtol=4e-4, atol=0))
    # print(np.testing.assert_allclose(u2_x.reshape(-1).detach().cpu().numpy(), u_x.reshape(-1).detach().cpu().numpy(), rtol=1.5e-1, atol=0))
    # print(np.testing.assert_allclose(u2_y.reshape(-1).detach().cpu().numpy(), u_y.reshape(-1).detach().cpu().numpy(), rtol=8e-2, atol=0))
    
    number = numb * n_cell * cell_dim
    # print(u_xx.reshape(-1)[8288269:8288269+100], u2_xx.reshape(-1)[8288269:8288269+100])
    # print(u_yy.reshape(-1)[6150152], u2_yy.reshape(-1)[6150152])
    # exit(1)
    
    print('val == val2, max_error: {} at {}'.format((val.reshape(-1)-val2.reshape(-1)).abs().max(),(val.reshape(-1)-val2.reshape(-1)).abs().argmax()))
    print('u_cell == u2_cell, max_error: {} at {}'.format((u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().max(),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().argmax()))
    print('u_x == u2_x, max_error: {} at {}'.format((u_x.reshape(-1)-u2_x.reshape(-1)).abs().max(),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().argmax()))
    print('u_y == u2_y, max_error: {} at {}'.format((u_y.reshape(-1)-u2_y.reshape(-1)).abs().max(),(u_y.reshape(-1)-u2_y.reshape(-1)).abs().argmax()))

    print('u_xx == u2_xx, max_error: {} at {}'.format((u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().max(),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().argmax()))
    print('u_yy == u2_yy, may_error: {} at {}'.format((u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().max(),(u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().argmax()))
    # print('u_cell_x == u2_cell_x, max_error: {} at {}'.format((u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().max(),(u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().argmax()))
    # print('u_cell_y == u2_cell_y, max_error: {} at {}'.format((u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().max(),(u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().argmax()))
    ''' ok before here'''
    print('u_x_cell == u2_x_cell, max_error: {} at {}'.format((u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().max(),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().argmax()))
    print('u_y_cell == u2_y_cell, max_error: {} at {}'.format((u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().max(),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().argmax()))
    # print('u_x_y == u2_x_y, max_error: {} at {}'.format((u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().max(),(u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().argmax()))
    # print('u_y_x == u2_y_x, max_error: {} at {}'.format((u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().max(),(u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().argmax()))

    # print('u_grid == u2_grid, max_error: {} at {}'.format(((u_grid.reshape(-1)-u2_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().max(),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().argmax()))
    # print('u_grid_grid == u2_grid_grid, max_error: {} at {}'.format(((u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().max(),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().argmax()))
    # print('u_input_grid == u2_input_grid, max_error: {} at {}'.format(((u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().max(),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().argmax()))
    # print('u_grid_input == u2_grid_input, max_error: {} at {}'.format(((u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().max(),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().argmax()))

    print('u_xx_cell == u2_xx_cell, max_error: {} at {}'.format((u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().max(),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().argmax()))
    print('u_yy_cell == u2_yy_cell, max_error: {} at {}'.format((u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().max(),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().argmax()))
    # print('u_xxx == u2_xxx: {}, max_error: {} at {}'.format(((u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs()<1e-4).sum()==(number),(u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs().max(),(u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs().argmax()))
    # print('u_yyy == u2_yyy: {}, max_error: {} at {}'.format(((u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs()<1e-4).sum()==(number),(u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs().max(),(u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs().argmax()))

    # torch.set_printoptions(threshold=100000*16*4)
    # different = u_xx.squeeze() - u2_xx.squeeze()

    f2_pred = u2_y*2 + 5*(val2.to('cuda')**3) - 5*val2.to('cuda') - 0.0001*u2_xx  # u2_yy + u2_xx + val2.to('cuda') #u2_y + val2.to('cuda')*u2_x -(0.01/np.pi)*u2_xx #
    loss2_f = torch.mean((f2_pred)**2)
    # loss2_u = torch.mean((val2_ic - ub)**2)
    
    # loss2_f = 0.0001 * loss2 + loss2_u

    print("----dloss2----")
    dloss2 = torch.autograd.grad(
        loss2_f, cells, 
        grad_outputs=torch.ones_like(loss2_f),
        retain_graph=True,
        create_graph=True
    )[0]
    # print("----ddloss2----")
    # ddloss2 = torch.autograd.grad(
    #     dloss2, cells, 
    #     grad_outputs=torch.ones_like(dloss2),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]


    f_pred = u_y*2 + 5*(val.to('cuda')**3) - 5*val.to('cuda') - 0.0001*u_xx  #u_yy + u_xx + val.to('cuda') #u_y + val.to('cuda')*u_x -(0.01/np.pi)*u_xx #
    loss_f = torch.mean((f_pred)**2)
    # loss_u = torch.mean((val_ic - ub)**2)
    # loss_f = 0.0001 * loss + loss_u

    dloss = torch.autograd.grad(
        loss_f, cells, 
        grad_outputs=torch.ones_like(loss_f),
        retain_graph=True,
        create_graph=True
    )[0]

    # ddloss = torch.autograd.grad(
    #     dloss, cells, 
    #     grad_outputs=torch.ones_like(dloss),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]

    # print(dloss2.shape)
    # print(dloss.shape)
    # dloss = dloss.flatten()
    # dloss2 = dloss2.flatten()
    print('dloss == dloss2: {}, max_error: {} at {}'.format(((dloss.reshape(-1)-dloss2.reshape(-1)).abs()<1e-4).sum()==(number),(dloss.reshape(-1)-dloss2.reshape(-1)).abs().max(),(dloss.reshape(-1)-dloss2.reshape(-1)).abs().argmax()))
    # print('ddloss == ddloss2: {}, max_error: {} at {}'.format(((ddloss.reshape(-1)-ddloss2.reshape(-1)).abs()<1e-4).sum()==(number),(ddloss.reshape(-1)-ddloss2.reshape(-1)).abs().max(),(ddloss.reshape(-1)-ddloss2.reshape(-1)).abs().argmax()))
    
    print('dloss error: ',np.testing.assert_allclose((dloss).reshape(-1).detach().cpu().numpy(), (dloss2).reshape(-1).detach().cpu().numpy(), rtol=1e-4, atol=0))

    # print(np.testing.assert_allclose(u2_xx_cell.reshape(-1).detach().cpu().numpy(), u_xx_cell.reshape(-1).detach().cpu().numpy(), rtol=1e-3, atol=0))
    # exit(1)