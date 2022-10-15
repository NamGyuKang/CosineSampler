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

def apply_cosine_step_mode_enum(apply_cosine_step):
    if apply_cosine_step == 'cosine':
        return True
    elif apply_cosine_step == 'bilinear':
        return False


class CosineSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode='zeros', align_corners = True, apply_cosine_step = 'cosine'):
        ''' offset merge kernel'''
        print('1st F')
        
        offset  =torch.linspace(0, 1-(1/input.shape[0]), input.shape[0]).to('cuda')
        ctx.padding_mode = padding_mode
        output = _cosine.forward(input, grid, offset, padding_mode_enum(padding_mode=padding_mode), align_corners, apply_cosine_step_mode_enum(apply_cosine_step=apply_cosine_step))
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        ctx.apply_cosine_step = apply_cosine_step

        return output

    @staticmethod
    def backward(ctx, gradOut):
        
        input, grid = ctx.saved_tensors
        print('1st B')
        if (gradOut == 0.).all().item():
            return torch.zeros_like(input), torch.zeros_like(grid), None, None, None

        d_input, d_grid = CosineSamplerBackward.apply(input, grid, gradOut.contiguous(), ctx.padding_mode, ctx.align_corners, ctx.apply_cosine_step)


        return d_input, d_grid, None, None, None


class CosineSamplerBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, gOut, padding_mode="zeros", align_corners=True, apply_cosine_step='cosine'):
        print('2st F')
        ctx.align_corners = align_corners
        ctx.apply_cosine_step = apply_cosine_step
        ctx.padding_mode = padding_mode

        offset = torch.linspace(0,1-(1/input.shape[0]),input.shape[0]).to("cuda")
        print('그라드', input.requires_grad)

        gInput, gGrid = _cosine.backward(gOut, input, grid, offset, padding_mode_enum(padding_mode),
                                            ctx.align_corners, apply_cosine_step_mode_enum(apply_cosine_step), input.requires_grad)
        
        ctx.save_for_backward(input, grid, gOut)

        return gInput, gGrid

    @staticmethod
    def backward(ctx, gOutInput, gOutGrid):
        print('2st B')
        input, grid, gOut = ctx.saved_tensors

        gInput, gGrid, ggOut = CosineSamplerBackwardBackward.apply(input, grid, gOut, gOutInput.contiguous(), gOutGrid.contiguous(), ctx.padding_mode, ctx.align_corners, ctx.apply_cosine_step)


        return gInput, gGrid, ggOut, None, None, None

class CosineSamplerBackwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, gOut, gOutInput, gOutGrid, padding_mode="zeros", align_corners=True, apply_cosine_step='cosine'):
        print('3st F')
        
        ctx.align_corners = align_corners
        ctx.apply_cosine_step = apply_cosine_step
        ctx.padding_mode = padding_mode

        offset = torch.linspace(0,1-(1/input.shape[0]),input.shape[0]).to("cuda")
        
        input_requires_grad = gOutInput is not None and (gOutInput != 0.).any().item()
        print('3F 그라드', input_requires_grad)

        gInput, gGrid, ggOut = _cosine.backward_backward(gOutInput, gOutGrid, input, grid, gOut, offset,
                                                                    padding_mode_enum(padding_mode), align_corners,
                                                                    apply_cosine_step_mode_enum(apply_cosine_step), input_requires_grad)
        ctx.save_for_backward(input, grid, gOut, gOutInput, gOutGrid, gGrid)

        return gInput, gGrid, ggOut

    @staticmethod
    def backward(ctx, gOutgInput, gOutgGrid, gOutggOut):
        print('3st B')
        align_corners = ctx.align_corners
        apply_cosine_step = ctx.apply_cosine_step
        padding_mode = ctx.padding_mode
        input, grid, gOut, gOutInput, gOutGrid, gGrid = ctx.saved_tensors

        offset = torch.linspace(0,1-(1/input.shape[0]),input.shape[0]).to("cuda")
        input_requires_grad = gOutgInput is not None and (gOutgInput != 0.).any().item()
        print('3B 그라드', input_requires_grad)
        # import numpy as np
        # print(np.testing.assert_allclose(gOutgGrid.reshape(-1).detach().cpu().numpy(), gOutGrid.reshape(-1).detach().cpu().numpy(), rtol=1e-2, atol=0))
        gInput, ggOut2 = _cosine.backward_backward_backward(input, grid, gOut, gOutggOut, gOutGrid, gOutgGrid.contiguous(), offset,
                                                                    padding_mode_enum(padding_mode), align_corners,
                                                                    apply_cosine_step_mode_enum(apply_cosine_step), input_requires_grad)
        

        return gInput, None, ggOut2, None, None, None, None, None


if __name__ == '__main__':
    torch.manual_seed(51)
    torch.cuda.manual_seed(51)
    
    off= True
    numb = 100000
    n_cell = 96
    cell_dim = 4
    step = 'cosine'
    padding_mode = 'zeros' # border, reflection
    align_corners = True

    cells = torch.nn.Parameter(torch.rand([n_cell, cell_dim, 16, 16], device = 'cuda')).requires_grad_(True)
    x = (torch.rand((n_cell, 1,numb, 1), dtype = torch.float32)).requires_grad_(True)
    y = (torch.rand((n_cell, 1,numb, 1), dtype = torch.float32)).requires_grad_(True)

    x = x*2-1
    y = y*2-1

    x = x.to('cuda')
    y = y.to('cuda')

    grid = torch.cat([y, x], -1)
    # with torch.no_grad():
    # grid = grid.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])
    # grid = grid.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])

    # grid = torch.rand((n_cell, 1, numb, 2), dtype = torch.float32).requires_grad_(True).to('cuda')
    
    start = time.time()

    val2 = CosineSampler.apply(cells, grid, padding_mode, align_corners, step)
    end = time.time()

    print('custom forward: ',end - start)
    net= []
    net.append(torch.nn.Linear(cell_dim, 16))
    net.append(torch.nn.Tanh())
    for i in range(2-2): 
        net.append(torch.nn.Linear(16, 16))
        net.append(torch.nn.Tanh())
    net.append(torch.nn.Linear(16, 1))


    # deploy layers
    net = torch.nn.Sequential(*net)
    
    val2 = val2.to("cpu")
    # val2 = torch.tanh(val2)

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


    # print(np.testing.assert_allclose(ddval.reshape(-1)tach().cpu().numpy(), u2_x.reshape(-1)tach().cpu().numpy(), rtol=1e-7, atol=0))
    # exit(1)

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


    print("--u2_cell_x--")
    u2_cell_x = torch.autograd.grad(
        u2_cell, x, 
        grad_outputs=torch.ones_like(u2_cell),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_cell_x.shape)

    print("--u2_cell_y--")
    u2_cell_y = torch.autograd.grad(
        u2_cell, y, 
        grad_outputs=torch.ones_like(u2_cell),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_cell_y.shape)

    print("--u2_x_cell--")
    u2_x_cell = torch.autograd.grad(
        u2_x, cells, 
        grad_outputs=torch.ones_like(u2_x),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_x_cell.shape)

    print("--u2_y_cell--")
    u2_y_cell = torch.autograd.grad(
        u2_y, cells, 
        grad_outputs=torch.ones_like(u2_y),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_y_cell.shape)

    print("--u2_x_y--")
    u2_x_y = torch.autograd.grad(
        u2_x, y, 
        grad_outputs=torch.ones_like(u2_x),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_x_y.shape)

    print("--u2_y_x--")
    u2_y_x = torch.autograd.grad(
        u2_y, x, 
        grad_outputs=torch.ones_like(u2_y),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_y_x.shape)

    # print("---u2grid---")
    # u2_grid = torch.autograd.grad(
    #     val2, grid, 
    #     grad_outputs=torch.ones_like(val2),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]


    # print("---u2gridgrid---")
    # u2_grid_grid = torch.autograd.grad(
    #     u2_grid, grid, 
    #     grad_outputs=torch.ones_like(u2_grid),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]

    # print("---u2inputgrid---")
    # u2_input_grid = torch.autograd.grad(
    #     u2_cell, grid, 
    #     grad_outputs=torch.ones_like(u2_cell),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]
    
    # print("---u2gridinput---")
    # u2_grid_input = torch.autograd.grad(
    #     u2_grid, cells, 
    #     grad_outputs=torch.ones_like(u2_grid),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]

    print("--u2_xx_cell--")
    u2_xx_cell = torch.autograd.grad(
        u2_xx, cells, 
        grad_outputs=torch.ones_like(u2_xx),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u2_xx_cell.shape)

    print("--u2_yy_cell--")
    u2_yy_cell = torch.autograd.grad(
        u2_yy, cells, 
        grad_outputs=torch.ones_like(u2_yy),
        retain_graph=True,
        create_graph=True
    )[0]
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
    end = time.time()
    print('python forward: ', end - start)



    val = val.to("cpu")
    # val = torch.tanh(val)
    # # val = torch.tanh(val)
    
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

    u_y = torch.autograd.grad(
        val, y, 
        grad_outputs=torch.ones_like(val),
        retain_graph=True,
        create_graph=True
    )[0]

    print("--u_xx--")
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


    print("--u_cell_x--")
    u_cell_x = torch.autograd.grad(
        u_cell, x, 
        grad_outputs=torch.ones_like(u_cell),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_cell_x.shape)

    print("--u_cell_y--")
    u_cell_y = torch.autograd.grad(
        u_cell, y, 
        grad_outputs=torch.ones_like(u_cell),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_cell_y.shape)

    print("--u_x_cell--")
    u_x_cell = torch.autograd.grad(
        u_x, cells, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_x_cell.shape)

    print("--u_y_cell--")
    u_y_cell = torch.autograd.grad(
        u_y, cells, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_y_cell.shape)

    print("--u_x_y--")
    u_x_y = torch.autograd.grad(
        u_x, y, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_x_y.shape)

    print("--u_y_x--")
    u_y_x = torch.autograd.grad(
        u_y, x, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_y_x.shape)


    # print("---ugrid---")
    # u_grid = torch.autograd.grad(
    #     val, grid, 
    #     grad_outputs=torch.ones_like(val),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]


    # print("---ugridgrid---")
    # u_grid_grid = torch.autograd.grad(
    #     u_grid, grid, 
    #     grad_outputs=torch.ones_like(u_grid),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]

    # print("---uinputgrid---")
    # u_input_grid = torch.autograd.grad(
    #     u_cell, grid, 
    #     grad_outputs=torch.ones_like(u_cell),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]
    
    # print("---ugridinput---")
    # u_grid_input = torch.autograd.grad(
    #     u_grid, cells, 
    #     grad_outputs=torch.ones_like(u_grid),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]

    print("--u_xx_cell--")
    u_xx_cell = torch.autograd.grad(
        u_xx, cells, 
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_xx_cell.shape)

    print("--u_yy_cell--")
    u_yy_cell = torch.autograd.grad(
        u_yy, cells, 
        grad_outputs=torch.ones_like(u_yy),
        retain_graph=True,
        create_graph=True
    )[0]
    print("shape check", u_yy_cell.shape)

    # print("--u_xxx--")
    # u_xxx = torch.autograd.grad(
    #     u_xx, x, 
    #     grad_outputs=torch.ones_like(u_xx),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]
    # print("shape check", u_xxx.shape)

    # print("--u_yyy--")
    # u_yyy = torch.autograd.grad(
    #     u_yy, y, 
    #     grad_outputs=torch.ones_like(u_yy),
    #     retain_graph=True,
    #     create_graph=True
    # )[0]
    # print("shape check", u_yyy.shape)

    # u2_cell/=96
    # u2_x/=96
    # u2_y/=96 
    # u2_xx/=96; u2_yy/=96;  
    # u2_cell_x/=96; u2_cell_y/=96; u2_x_cell/=96; u2_y_cell/=96; u2_x_y/=96; u2_y_x/=96; 
    # u2_xx_cell/=96; u2_yy_cell/=96
    # u_cell/=96
    # u_x/=96
    # u_y/=96 
    # u_xx/=96; u_yy/=96;  
    # u_cell_x/=96; u_cell_y/=96; u_x_cell/=96; u_y_cell/=96; u_x_y/=96; u_y_x/=96; 
    # u_xx_cell/=96; u_yy_cell/=96

    import numpy as np    
    # print(np.testing.assert_allclose(u2_xx.reshape(-1).detach().cpu().numpy(), u_xx.reshape(-1).detach().cpu().numpy(), rtol=1e-2, atol=0))
    # print(np.testing.assert_allclose(u2_yy_cell.reshape(-1).detach().cpu().numpy(), u_yy_cell.reshape(-1).detach().cpu().numpy(), rtol=4e-4, atol=0))
    # print(np.testing.assert_allclose(u2_x.reshape(-1).detach().cpu().numpy(), u_x.reshape(-1).detach().cpu().numpy(), rtol=1.5e-1, atol=0))
    # print(np.testing.assert_allclose(u2_y.reshape(-1).detach().cpu().numpy(), u_y.reshape(-1).detach().cpu().numpy(), rtol=8e-2, atol=0))
    
    number = numb * n_cell * cell_dim
    # print(u_xx.reshape(-1)[8288269:8288269+100], u2_xx.reshape(-1)[8288269:8288269+100])
    # print(u_yy.reshape(-1)[6150152], u2_yy.reshape(-1)[6150152])
    # exit(1)
    
    print('val == val2: {}, max_error: {} at {}'.format(((val.reshape(-1)-val2.reshape(-1)).abs()<1e-4).sum()==(number),(val.reshape(-1)-val2.reshape(-1)).abs().max(),(val.reshape(-1)-val2.reshape(-1)).abs().argmax()))
    print('u_cell == u2_cell: {}, max_error: {} at {}'.format(((u_cell.reshape(-1)-u2_cell.reshape(-1)).abs()<1e-4).sum()==(number),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().max(),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().argmax()))
    print('u_x == u2_x: {}, max_error: {} at {}'.format(((u_x.reshape(-1)-u2_x.reshape(-1)).abs()<1e-4).sum()==(number),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().max(),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().argmax()))
    print('u_y == u2_y: {}, max_error: {} at {}'.format(((u_y.reshape(-1)-u2_y.reshape(-1)).abs()<1e-4).sum()==(number),(u_y.reshape(-1)-u2_y.reshape(-1)).abs().max(),(u_y.reshape(-1)-u2_y.reshape(-1)).abs().argmax()))

    print('u_xx == u2_xx: {}, max_error: {} at {}'.format(((u_xx.reshape(-1)-u2_xx.reshape(-1)).abs()<1e-4).sum()==(number),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().max(),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().argmax()))
    print('u_yy == u2_yy: {}, may_error: {} at {}'.format(((u_yy.reshape(-1)-u2_yy.reshape(-1)).abs()<1e-4).sum()==(number),(u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().max(),(u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().argmax()))
    print('u_cell_x == u2_cell_x: {}, max_error: {} at {}'.format(((u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs()<1e-4).sum()==(number),(u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().max(),(u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().argmax()))
    print('u_cell_y == u2_cell_y: {}, max_error: {} at {}'.format(((u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs()<1e-4).sum()==(number),(u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().max(),(u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().argmax()))
    ''' ok before here'''
    print('u_x_cell == u2_x_cell: {}, max_error: {} at {}'.format(((u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs()<1e-4).sum()==(number),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().max(),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().argmax()))
    print('u_y_cell == u2_y_cell: {}, max_error: {} at {}'.format(((u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs()<1e-4).sum()==(number),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().max(),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().argmax()))
    print('u_x_y == u2_x_y: {}, max_error: {} at {}'.format(((u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs()<1e-4).sum()==(number),(u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().max(),(u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().argmax()))
    print('u_y_x == u2_y_x: {}, max_error: {} at {}'.format(((u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs()<1e-4).sum()==(number),(u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().max(),(u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().argmax()))

    # print('u_grid == u2_grid: {}, max_error: {} at {}'.format(((u_grid.reshape(-1)-u2_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().max(),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().argmax()))
    # print('u_grid_grid == u2_grid_grid: {}, max_error: {} at {}'.format(((u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().max(),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().argmax()))
    # print('u_input_grid == u2_input_grid: {}, max_error: {} at {}'.format(((u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().max(),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().argmax()))
    # print('u_grid_input == u2_grid_input: {}, max_error: {} at {}'.format(((u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().max(),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().argmax()))

    print('u_xx_cell == u2_xx_cell: {}, max_error: {} at {}'.format(((u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs()<1e-4).sum()==(number),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().max(),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().argmax()))
    print('u_yy_cell == u2_yy_cell: {}, max_error: {} at {}'.format(((u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs()<1e-4).sum()==(number),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().max(),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().argmax()))
    # print('u_xxx == u2_xxx: {}, max_error: {} at {}'.format(((u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs()<1e-4).sum()==(number),(u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs().max(),(u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs().argmax()))
    # print('u_yyy == u2_yyy: {}, max_error: {} at {}'.format(((u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs()<1e-4).sum()==(number),(u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs().max(),(u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs().argmax()))

    torch.set_printoptions(threshold=100000*16*4)
    # different = u_xx.squeeze() - u2_xx.squeeze()

    f2_pred =  u2_y + val2.to('cuda') * u2_x - u2_xx
    loss2_f = torch.mean(f2_pred**2)

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


    f_pred =  u_y + val.to('cuda') * u_x - u_xx
    loss_f = torch.mean(f_pred**2)

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

    print(np.testing.assert_allclose((dloss).reshape(-1).detach().cpu().numpy(), (dloss2).reshape(-1).detach().cpu().numpy(), rtol=1e-2, atol=0))

    exit(1)