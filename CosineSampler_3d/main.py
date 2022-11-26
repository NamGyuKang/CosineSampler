import torch
from torch.utils.cpp_extension import load
# import grid_sample_temp
# import os
# import numpy as np
# import time


_cosine = load(name="_cosine", sources = ['/hdd/kng/CosineSampler/CosineSampler_3d/cosine_sampler_3d_kernel.cu', '/hdd/kng/CosineSampler/CosineSampler_3d/cosine_sampler_3d.cpp'])
def padding_mode_enum(padding_mode):
    if padding_mode == "zeros":
        return 0
    elif padding_mode == "border":
        return 1
    else:  # padding_mode == 'reflection'
        return 2

def kernel_enum(kernel):
    if kernel == 'cosine':
        return 0
    elif kernel == 'trilinear':
        return 1
    elif kernel == 'smooth-step':
        return 2

class CosineSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode="zeros", align_corners=True, kernel='cosine', multicell = True):
        if multicell:
            ctx.offset  =torch.linspace(0, 1-(1/input.shape[0]), input.shape[0]).to('cuda')    
        else:
            ctx.offset = torch.zeros(input.shape[0]).to('cuda')   

        output = _cosine.forward(input, grid, ctx.offset, padding_mode_enum(padding_mode), align_corners, kernel_enum(kernel), multicell)
        ctx.save_for_backward(input, grid)
        
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        ctx.kernel = kernel
        ctx.multicell = multicell
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, grid = ctx.saved_tensors
        
        if (grad_out == 0.).all().item():
            return torch.zeros_like(input), torch.zeros_like(grid), None, None, None, None, None
         
        d_input, d_grid = CosineSamplerBackward.apply(input, grid, grad_out.contiguous(), ctx.offset, ctx.padding_mode, ctx.align_corners, ctx.kernel, ctx.multicell) 
        return d_input, d_grid, None, None, None, None

class CosineSamplerBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, grad_out, offset, padding_mode="zeros", align_corners=True, kernel = 'cosine', multicell = True):
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        ctx.offset = offset
        ctx.kernel = kernel
        ctx.multicell = multicell
        grad_input, grad_grid = _cosine.backward(grad_out, input, grid, offset, padding_mode_enum(padding_mode),
                                            ctx.align_corners, input.requires_grad, kernel_enum(kernel), multicell)
        ctx.save_for_backward(input, grid, grad_out)
        
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, gOutInput, gOutGrid):
        input, grid, gOut = ctx.saved_tensors

        gInput, gGrid, ggOut = CosineSamplerBackwardBackward.apply(input, grid, gOut, gOutInput.contiguous(), gOutGrid.contiguous(), ctx.offset, ctx.padding_mode, ctx.align_corners, ctx.kernel, ctx.multicell)


        return gInput, gGrid, ggOut, None, None, None, None, None
    


class CosineSamplerBackwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, gOut, gOutInput, gOutGrid, offset, padding_mode="zeros", align_corners=True, kernel = 'cosine', multicell = True):
        
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        ctx.offset = offset
        ctx.kernel = kernel
        ctx.multicell = multicell

        
        input_requires_grad = gOutInput is not None and (gOutInput != 0.).any().item()

        gInput, gGrid, ggOut = _cosine.backward_backward(gOutInput, gOutGrid, input, grid, gOut, offset,
                                                                    padding_mode_enum(padding_mode), align_corners, input_requires_grad, kernel_enum(kernel), multicell)
        ctx.save_for_backward(input, grid, gOut, gOutGrid)
        
        return gInput, gGrid, ggOut

    @staticmethod
    def backward(ctx, gOutgInput, gOutgGrid, gOutggOut):
        align_corners = ctx.align_corners
        padding_mode = ctx.padding_mode
        input, grid, gOut, gOutGrid,  = ctx.saved_tensors 
        

        input_requires_grad = gOutgInput is not None and (gOutgInput != 0.).any().item()
        gInput, ggOut = _cosine.backward_backward_backward(input, grid, gOut, gOutGrid, gOutgGrid.contiguous(), ctx.offset,
                                                                    padding_mode_enum(padding_mode), align_corners, input_requires_grad, kernel_enum(ctx.kernel), ctx.multicell) 

        b_input, _, _ = CosineSamplerBackwardBackward.apply(input, grid, gOutggOut.contiguous(), torch.ones_like(gOutgInput), gOutGrid, ctx.offset, padding_mode, align_corners, ctx.kernel, ctx.multicell)

        return gInput+b_input, None, ggOut, None, None, None, None, None, None, None

   

    
# if __name__ == "__main__":
#     torch.manual_seed(6)
#     torch.cuda.manual_seed(6)
    
#     for padding_mode in ["zeros", "border", "reflection"]:
#         for align_corners in [True, False]:
#             # input = (torch.rand([13,4,16,16,16], device="cuda")).requires_grad_(True)
#             # grid = (torch.rand([13,1,1,100,3], device="cuda") * 2. - 1.).requires_grad_(True)
    
#             # # CosineSampler forward vs native forward
#             # out1 = CosineSampler.apply(input, grid, padding_mode, align_corners, True)
#             # # out2 = torch.nn.functional.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
#             # out3 = grid_sample_temp.grid_sample_3d(input, grid, step='Cosinestep', offset=False).unsqueeze(2)
#             # # print(out1.shape, out3.shape)
#             # # print(np.testing.assert_allclose(out1.reshape(-1).detach().cpu().numpy(), out3.reshape(-1).detach().cpu().numpy(), rtol=1e-4, atol=0))
#             # # assert torch.allclose(out1, out3)

#             # # CosineSampler backward vs native backward
#             # grad1_input, grad1_grid = torch.autograd.grad(out1, [input, grid], torch.ones_like(out1), create_graph=True)
#             # grad3_input, grad3_grid = torch.autograd.grad(out3, [input, grid], torch.ones_like(out3), create_graph=True)
            
#             # u_c_c, u_c_xy = torch.autograd.grad(grad1_input, [input, grid], torch.ones_like(grad1_input), retain_graph= True, create_graph=True)
#             # u_xy_c, u_xy_xy = torch.autograd.grad(grad1_grid, [input, grid] , torch.ones_like(grad1_grid), retain_graph= True, create_graph=True)
            
#             # u2_c_xy = torch.autograd.grad(grad3_input, grid, torch.ones_like(grad3_input), retain_graph= True, create_graph=True)[0]
#             # u2_xy_c, u2_xy_xy = torch.autograd.grad(grad3_grid, [input, grid] , torch.ones_like(grad3_grid), retain_graph= True, create_graph=True)
            
#             # print(np.testing.assert_allclose(u_xy_cosine.reshape(-1).detach().cpu().numpy(), u2_xy_cosine.reshape(-1).detach().cpu().numpy(), rtol=1e-7, atol=0))
#             # exit(1)


#             point_num = 100000
#             n_cell = 50
#             cell_dim = 4
#             step = True

#             # cells = (torch.rand([n_cell,cell_dim,16,16,16], device="cuda")).requires_grad_(True)
#             # cells = (torch.rand([n_cell,cell_dim,16,16,16], device="cuda")).requires_grad_(True)
#             cells = torch.nn.Parameter(torch.rand([n_cell, cell_dim, 16, 16, 16], device = 'cuda')).requires_grad_(True)
#             # cells = cells.data.uniform_(-1e-05, 1e-05).requires_grad_(True)
    
#             yxz = np.random.rand(point_num, 3)
#             # yxz[..., 1] = yxz[..., 1]*2-1
#             yxz_f = torch.tensor(yxz, requires_grad=True).float()
#             y = yxz_f[:, 0:1]
#             x = yxz_f[:, 1:2]
#             z = yxz_f[:, 2:3]

#             # x = torch.rand((point_num, 1), dtype=torch.float32) *2-1
#             # y = torch.rand((point_num, 1), dtype=torch.float32) *2-1
#             # z = torch.rand((point_num, 1), dtype=torch.float32) *2-1

#             # x.requires_grad = True
#             # y.requires_grad = True
#             # z.requires_grad = True

#             # x2 = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([n_cell,1, 1,1,1])
#             # y2 = y.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([n_cell,1, 1,1,1])
#             # z2 = z.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([n_cell,1, 1,1,1])
#             x = x.to("cuda")
#             y = y.to("cuda")
#             z = z.to("cuda")

#             grid = torch.cat([z, y, x], -1)
#             grid = grid.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1, 1])

#             # grid = (torch.rand([1,1,10000,2], device="cuda") * 2. - 1.).repeat([n_cell, 1, 1, 1]).requires_grad_(True)
#             start = time.time()
#             val2 = CosineSampler.apply(cells, grid, padding_mode, align_corners, 'cosine', False)
#             end = time.time()
#             print("cuda: ", end - start)
#             # out2 = torch.nn.functional.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
            
#             net= []
#             net.append(torch.nn.Linear(4, 16))
#             net.append(torch.nn.Tanh())
#             for i in range(2-2): 
#                 net.append(torch.nn.Linear(16, 16))
#                 net.append(torch.nn.Tanh())
#             net.append(torch.nn.Linear(16, 1))


#             # deploy layers
#             net = torch.nn.Sequential(*net)




#             val2 = val2.to("cpu")
#             # val2 = torch.tanh(val2)

#             val2 = val2.sum(0).view(cell_dim,-1).t()
#             val2 = net(val2)        

#             print("---u2cell---")
#             start = time.time()
#             u2_cell = torch.autograd.grad(
#                 val2, cells, 
#                 grad_outputs=torch.ones_like(val2),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2cell cuda: ", end - start)
            


#             print("---u2x---")
#             start = time.time()
#             u2_x = torch.autograd.grad(
#                 val2, x, 
#                 grad_outputs=torch.ones_like(val2),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2x cuda: ", end - start)
            


#             # print(np.testing.assert_allclose(ddval.squeeze().detach().cpu().numpy(), u2_x.squeeze().detach().cpu().numpy(), rtol=1e-7, atol=0))
#             # exit(1)

#             print("---u2y==--")
#             start = time.time()
#             u2_y = torch.autograd.grad(
#                 val2, y, 
#                 grad_outputs=torch.ones_like(val2),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2y cuda: ", end - start)
            
#             print("---u2z==--")
#             start = time.time()
#             u2_z = torch.autograd.grad(
#                 val2, z, 
#                 grad_outputs=torch.ones_like(val2),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2z cuda: ", end - start)
            

#             print("---u2xx==--")
#             start = time.time()
#             u2_xx = torch.autograd.grad(
#                 u2_x, x, 
#                 grad_outputs=torch.ones_like(u2_x),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2xx cuda: ", end - start)
            


#             print("--u2_yy--")
#             start = time.time()
#             u2_yy = torch.autograd.grad(
#                 u2_y, y, 
#                 grad_outputs=torch.ones_like(u2_y),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2yy cuda: ", end - start)
            

#             print("--u2_zz--")
#             start = time.time()
#             u2_zz = torch.autograd.grad(
#                 u2_z, z, 
#                 grad_outputs=torch.ones_like(u2_z),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2zz cuda: ", end - start)
            




#             print("--u2_x_cell--")
#             start = time.time()
#             u2_x_cell = torch.autograd.grad(
#                 u2_x, cells, 
#                 grad_outputs=torch.ones_like(u2_x),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2x_cell cuda: ", end - start)
#             # print("shape check", u2_x_cell.shape)
            

#             print("--u2_y_cell--")
#             start = time.time()
#             u2_y_cell = torch.autograd.grad(
#                 u2_y, cells, 
#                 grad_outputs=torch.ones_like(u2_y),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u2_y_cell.shape)
#             print("u2y_cell cuda: ", end - start)
            
#             print("--u2_z_cell--")
#             start = time.time()
#             u2_z_cell = torch.autograd.grad(
#                 u2_z, cells, 
#                 grad_outputs=torch.ones_like(u2_z),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u2z_cell cuda: ", end - start)
#             # shape check", u2_z_cell.shape)

#             print("--u2_xx_cell--")
#             start = time.time()
#             u2_xx_cell = torch.autograd.grad(
#                 u2_xx, cells, 
#                 grad_outputs=torch.ones_like(u2_xx),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u2_xx_cell.shape)
#             print("u2xx_cell cuda: ", end - start)
            
#             print("--u2_yy_cell--")
#             start = time.time()
#             u2_yy_cell = torch.autograd.grad(
#                 u2_yy, cells, 
#                 grad_outputs=torch.ones_like(u2_yy),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u2_yy_cell.shape)
#             print("u2yy_cell cuda: ", end - start)
            

#             print("--u2_zz_cell--")
#             start = time.time()
#             u2_zz_cell = torch.autograd.grad(
#                 u2_zz, cells, 
#                 grad_outputs=torch.ones_like(u2_zz),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u2_zz_cell.shape)
#             print("u2zz_cell cuda: ", end - start)
            


#             # print("--u2_x_y--")
#             # u2_x_y = torch.autograd.grad(
#             #     u2_x, y, 
#             #     grad_outputs=torch.ones_like(u2_x),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
#             # print("shape check", u2_x_y.shape)

#             # print("--u2_y_x--")
#             # u2_y_x = torch.autograd.grad(
#             #     u2_y, x, 
#             #     grad_outputs=torch.ones_like(u2_y),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
#             # print("shape check", u2_y_x.shape)

#             # print("---u2grid---")
#             # u2_grid = torch.autograd.grad(
#             #     val2, grid, 
#             #     grad_outputs=torch.ones_like(val2),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]


#             # print("---u2gridgrid---")
#             # u2_grid_grid = torch.autograd.grad(
#             #     u2_grid, grid, 
#             #     grad_outputs=torch.ones_like(u2_grid),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]

#             # print("---u2inputgrid---")
#             # u2_input_grid = torch.autograd.grad(
#             #     u2_cell, grid, 
#             #     grad_outputs=torch.ones_like(u2_cell),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
            
#             # print("---u2gridinput---")
#             # u2_grid_input = torch.autograd.grad(
#             #     u2_grid, cells, 
#             #     grad_outputs=torch.ones_like(u2_grid),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
#             # make_dot(val2, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val", format="png")
#             # make_dot(u2_x, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val_x", format="png")
#             # make_dot(u2_xx, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val_xx", format="png")
#             print("----")


#             # grid = torch.cat([y, x], dim=-1)#.unsqueeze(0).unsqueeze(0)


#             # grid = grid.repeat([cells.shape[0],1,1,1])
#             # print(grid.shape)
#             # exit(1)
#             # grid = torch.cat([z, y, x], -1)
#             # grid = grid.unsqueeze(0).unsqueeze(0).repeat([n_cell, 1, 1, 1])
#             start = time.time()
#             val = grid_sample_temp.grid_sample_3d(cells, grid, step='cosine', offset=False).unsqueeze(2)
#             end = time.time()
#             print("pytorch: ", end - start)
#             # val = grid_sample_temp.grid_sample_3d(cells, grid, step=step, offset=False).unsqueeze(2)
#             print(val.shape, 'val shape')
            
#             # val = grid_sample_temp.grid_sample_2d(cells, grid, step='cosine', offset=off)
#             val = val.to("cpu")
#             # val = torch.tanh(val)
#             # val = torch.tanh(val)
#             val = val.sum(0).view(cell_dim,-1).t()
#             val = net(val)
#             start = time.time()
#             u_cell = torch.autograd.grad(
#                 val, cells, 
#                 grad_outputs=torch.ones_like(val),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_cell pytorch: ", end - start)
            
#             start = time.time()
#             u_x = torch.autograd.grad(
#                 val, x, 
#                 grad_outputs=torch.ones_like(val),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_x pytorch: ", end - start)
            
#             start = time.time()
#             u_xx = torch.autograd.grad(
#                 u_x, x, 
#                 grad_outputs=torch.ones_like(u_x),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_xx pytorch: ", end - start)

#             start = time.time()
#             u_y = torch.autograd.grad(
#                 val, y, 
#                 grad_outputs=torch.ones_like(val),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_y pytorch: ", end - start)

#             start = time.time()
#             u_yy = torch.autograd.grad(
#                 u_y, y, 
#                 grad_outputs=torch.ones_like(u_y),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_yy pytorch: ", end - start)

#             start = time.time()
#             u_z = torch.autograd.grad(
#                 val, z, 
#                 grad_outputs=torch.ones_like(val),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_z pytorch: ", end - start)

#             start = time.time()
#             u_zz = torch.autograd.grad(
#                 u_z, z, 
#                 grad_outputs=torch.ones_like(u_z),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             print("u_zz pytorch: ", end - start)
            


#             print("--u_x_cell--")
#             start = time.time()
#             u_x_cell = torch.autograd.grad(
#                 u_x, cells, 
#                 grad_outputs=torch.ones_like(u_x),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u_x_cell.shape)
#             print("u_x_cell pytorch: ", end - start)
            

#             print("--u_y_cell--")
#             start = time.time()
#             u_y_cell = torch.autograd.grad(
#                 u_y, cells, 
#                 grad_outputs=torch.ones_like(u_y),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u_y_cell.shape)
#             print("u_y_cell pytorch: ", end - start)
            

#             print("--u_z_cell--")
#             start = time.time()
#             u_z_cell = torch.autograd.grad(
#                 u_z, cells, 
#                 grad_outputs=torch.ones_like(u_z),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u_z_cell.shape)
#             print("u_z_cell pytorch: ", end - start)
            

#             # print("--u_x_y--")
#             # u_x_y = torch.autograd.grad(
#             #     u_x, y, 
#             #     grad_outputs=torch.ones_like(u_x),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
#             # print("shape check", u_x_y.shape)

#             # print("--u_y_x--")
#             # u_y_x = torch.autograd.grad(
#             #     u_y, x, 
#             #     grad_outputs=torch.ones_like(u_y),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
#             # print("shape check", u_y_x.shape)


#             # print("---ugrid---")
#             # u_grid = torch.autograd.grad(
#             #     val, grid, 
#             #     grad_outputs=torch.ones_like(val),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]


#             # print("---ugridgrid---")
#             # u_grid_grid = torch.autograd.grad(
#             #     u_grid, grid, 
#             #     grad_outputs=torch.ones_like(u_grid),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]

#             # print("---uinputgrid---")
#             # u_input_grid = torch.autograd.grad(
#             #     u_cell, grid, 
#             #     grad_outputs=torch.ones_like(u_cell),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]
            
#             # print("---ugridinput---")
#             # u_grid_input = torch.autograd.grad(
#             #     u_grid, cells, 
#             #     grad_outputs=torch.ones_like(u_grid),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]

#             print("--u_xx_cell--")
#             u_xx_cell = torch.autograd.grad(
#                 u_xx, cells, 
#                 grad_outputs=torch.ones_like(u_xx),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u_xx_cell.shape)
#             print("u_xx_cell pytorch: ", end - start)
            

#             print("--u_yy_cell--")
#             u_yy_cell = torch.autograd.grad(
#                 u_yy, cells, 
#                 grad_outputs=torch.ones_like(u_yy),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u_yy_cell.shape)
#             print("u_yy_cell pytorch: ", end - start)
            

#             print("--u_zz_cell--")
#             u_zz_cell = torch.autograd.grad(
#                 u_zz, cells, 
#                 grad_outputs=torch.ones_like(u_zz),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             end = time.time()
#             # print("shape check", u_zz_cell.shape)
#             print("u_zz_cell pytorch: ", end - start)
            

#             # print(u_x_cells.shape)
#             # exit(1)
#             # u_cell = u_cell.flatten()
#             # u2_cell = u2_cell.flatten()

#             # u_x_cell = u_x_cell.flatten()
#             # u2_x_cell = u2_x_cell.flatten()

#             # u_y_cell = u_y_cell.flatten()
#             # u2_y_cell = u2_y_cell.flatten()

#             # u_cell_x = u_cell_x.flatten()
#             # u2_cell_x = u2_cell_x.flatten()

#             # u_cell_y = u_cell_y.flatten()
#             # u2_cell_y = u2_cell_y.flatten()

            

#             # u_grid = u_grid.flatten()
#             # u2_grid = u2_grid.flatten()

#             # u_grid_grid = u_grid_grid.flatten()
#             # u2_grid_grid = u2_grid_grid.flatten()


#             # u_grid_input = u_grid_input.flatten()
#             # u2_grid_input = u2_grid_input.flatten()

#             # u_input_grid = u_input_grid.flatten()
#             # u2_input_grid = u2_input_grid.flatten()            
            

#             # u_xx_cell = u_xx_cell.flatten()
#             # u2_xx_cell = u2_xx_cell.flatten()

#             # u_yy_cell = u_yy_cell.flatten()
#             # u2_yy_cell = u2_yy_cell.flatten()


#             # u_xx /= 640000 
#             # u_yy /= 640000

#             # u2_xx *= 640000 
#             # u2_yy *= 640000
#             # # u2_x_cell *= 6400
#             # # u2_y_cell *= 6400
#             # u2_x_y *= 640000
#             # u2_y_x *= 640000
#             # # u_x_cells = u_x_cells.flatten()
#             # # u2_x_cells = u2_x_cells.flatten()
#             # # u2_yy *= 2

#             # print(u_x_cells)
#             # exit(1)
#             # print(u_yy)
#             # print(u2_yy)
#             # print(np.testing.assert_allclose(u_xx.squeeze().detach().cpu().numpy(), u2_xx.squeeze().detach().cpu().numpy(), rtol=1e-4, atol=0))
#             # # # # # # # # # # # print(np.testing.assert_allclose(u2_xx.squeeze().detach().cpu().numpy(), u_xx.squeeze().detach().cpu().numpy(), rtol=1e-4, atol=0))
#             # exit(1)
#             import numpy as np    
#             # print(np.testing.assert_allclose(u2_yy_cell.reshape(-1).detach().cpu().numpy(), u_yy_cell.reshape(-1).detach().cpu().numpy(), rtol=4e-4, atol=0))
#             # print(np.testing.assert_allclose(u2_x.reshape(-1).detach().cpu().numpy(), u_x.reshape(-1).detach().cpu().numpy(), rtol=1.5e-1, atol=0))
#             # print(np.testing.assert_allclose(u2_y.reshape(-1).detach().cpu().numpy(), u_y.reshape(-1).detach().cpu().numpy(), rtol=8e-2, atol=0))
            
#             # number = point_num * n_cell * cell_dim
#             # print(u_xx.reshape(-1)[8288269:8288269+100], u2_xx.reshape(-1)[8288269:8288269+100])
#             # print(u_yy.reshape(-1)[6150152], u2_yy.reshape(-1)[6150152])
#             # exit(1)
            
#             print('val == val2, max_error: {} at {}'.format((val.reshape(-1)-val2.reshape(-1)).abs().max(),(val.reshape(-1)-val2.reshape(-1)).abs().argmax()))
#             print('u_cell == u2_cell, max_error: {} at {}'.format((u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().max(),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().argmax()))
#             print('u_x == u2_x, max_error: {} at {}'.format((u_x.reshape(-1)-u2_x.reshape(-1)).abs().max(),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().argmax()))
#             print('u_y == u2_y, max_error: {} at {}'.format((u_y.reshape(-1)-u2_y.reshape(-1)).abs().max(),(u_y.reshape(-1)-u2_y.reshape(-1)).abs().argmax()))
#             print('u_z == u2_z, max_error: {} at {}'.format((u_z.reshape(-1)-u2_z.reshape(-1)).abs().max(),(u_z.reshape(-1)-u2_z.reshape(-1)).abs().argmax()))

#             print('u_xx == u2_xx, max_error: {} at {}'.format((u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().max(),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().argmax()))
#             print('u_yy == u2_yy, may_error: {} at {}'.format((u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().max(),(u_yy.reshape(-1)-u2_yy.reshape(-1)).abs().argmax()))
#             print('u_zz == u2_zz, may_error: {} at {}'.format((u_zz.reshape(-1)-u2_zz.reshape(-1)).abs().max(),(u_zz.reshape(-1)-u2_zz.reshape(-1)).abs().argmax()))
#             # print('u_cell_x == u2_cell_x, max_error: {} at {}'.format((u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().max(),(u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().argmax()))
#             # print('u_cell_y == u2_cell_y, max_error: {} at {}'.format((u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().max(),(u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().argmax()))
#             ''' ok before here'''
#             print('u_x_cell == u2_x_cell, max_error: {} at {}'.format((u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().max(),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().argmax()))
#             print('u_y_cell == u2_y_cell, max_error: {} at {}'.format((u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().max(),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().argmax()))
#             print('u_z_cell == u2_z_cell, max_error: {} at {}'.format((u_z_cell.reshape(-1)-u2_z_cell.reshape(-1)).abs().max(),(u_z_cell.reshape(-1)-u2_z_cell.reshape(-1)).abs().argmax()))
#             # print('u_x_y == u2_x_y, max_error: {} at {}'.format((u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().max(),(u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().argmax()))
#             # print('u_y_x == u2_y_x, max_error: {} at {}'.format((u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().max(),(u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().argmax()))

#             # print('u_grid == u2_grid, max_error: {} at {}'.format(((u_grid.reshape(-1)-u2_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().max(),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().argmax()))
#             # print('u_grid_grid == u2_grid_grid, max_error: {} at {}'.format(((u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().max(),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().argmax()))
#             # print('u_input_grid == u2_input_grid, max_error: {} at {}'.format(((u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().max(),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().argmax()))
#             # print('u_grid_input == u2_grid_input, max_error: {} at {}'.format(((u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().max(),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().argmax()))

#             print('u_xx_cell == u2_xx_cell, max_error: {} at {}'.format((u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().max(),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().argmax()))
#             print('u_yy_cell == u2_yy_cell, max_error: {} at {}'.format((u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().max(),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().argmax()))
#             print('u_zz_cell == u2_zz_cell, max_error: {} at {}'.format((u_zz_cell.reshape(-1)-u2_zz_cell.reshape(-1)).abs().max(),(u_zz_cell.reshape(-1)-u2_zz_cell.reshape(-1)).abs().argmax()))
#            # print('u_xxx == u2_xxx: {}, max_error: {} at {}'.format(((u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs()<1e-4).sum()==(number),(u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs().max(),(u_xxx.reshape(-1)-u2_xxx.reshape(-1)).abs().argmax()))
#             # print('u_yyy == u2_yyy: {}, max_error: {} at {}'.format(((u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs()<1e-4).sum()==(number),(u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs().max(),(u_yyy.reshape(-1)-u2_yyy.reshape(-1)).abs().argmax()))

#             # torch.set_printoptions(threshold=100000*16*4)
#             # different = u_xx.squeeze() - u2_xx.squeeze()
            
#             f2_pred = u2_xx + u2_yy + u2_zz + val2.to('cuda') #u2_y*2 + 5*(val2.to('cuda')**3) - 5*val2.to('cuda') - 0.0001*u2_xx  # u2_yy + u2_xx + val2.to('cuda') #u2_y + val2.to('cuda')*u2_x -(0.01/np.pi)*u2_xx #
#             loss2_f = torch.mean((f2_pred)**2)
#             # loss2_u = torch.mean((val2_ic - ub)**2)
            
#             # loss2_f = 0.0001 * loss2 + loss2_u

#             print("----dloss2----")
#             dloss2 = torch.autograd.grad(
#                 loss2_f, cells, 
#                 grad_outputs=torch.ones_like(loss2_f),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]
#             # print("----ddloss2----")
#             # ddloss2 = torch.autograd.grad(
#             #     dloss2, cells, 
#             #     grad_outputs=torch.ones_like(dloss2),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]


#             f_pred = u_xx + u_yy + u_zz + val.to('cuda') #u2_y*2 + 5*(val2.to('cuda')**3) - 5*val2.to('cuda') - 0.0001*u2_xx  # u2_yy + u2_xx + val2.to('cuda') #u2_y + val2.to('cuda')*u2_x -(0.01/np.pi)*u2_xx #
#             loss_f = torch.mean((f_pred)**2)
#             # loss_u = torch.mean((val_ic - ub)**2)
#             # loss_f = 0.0001 * loss + loss_u

#             dloss = torch.autograd.grad(
#                 loss_f, cells, 
#                 grad_outputs=torch.ones_like(loss_f),
#                 retain_graph=True,
#                 create_graph=True
#             )[0]

#             # ddloss = torch.autograd.grad(
#             #     dloss, cells, 
#             #     grad_outputs=torch.ones_like(dloss),
#             #     retain_graph=True,
#             #     create_graph=True
#             # )[0]

#             # print(dloss2.shape)
#             # print(dloss.shape)
#             # dloss = dloss.flatten()
#             # dloss2 = dloss2.flatten()
#             print('dloss == dloss2, max_error: {} at {}'.format((dloss.reshape(-1)-dloss2.reshape(-1)).abs().max(),(dloss.reshape(-1)-dloss2.reshape(-1)).abs().argmax()))
#             # print('ddloss == ddloss2: {}, max_error: {} at {}'.format(((ddloss.reshape(-1)-ddloss2.reshape(-1)).abs()<1e-4).sum()==(number),(ddloss.reshape(-1)-ddloss2.reshape(-1)).abs().max(),(ddloss.reshape(-1)-ddloss2.reshape(-1)).abs().argmax()))
            
#             print('dloss error: ',np.testing.assert_allclose((dloss).reshape(-1).detach().cpu().numpy(), (dloss2).reshape(-1).detach().cpu().numpy(), rtol=1e-4, atol=0))
#             #                             eps=1e-4, atol=1e-3, rtol=1e-2)
#             exit(1)