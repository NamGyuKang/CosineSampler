import torch
from torch import autograd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
# from torchviz import make_dot
import grid_sample_temp
import time
from torch.utils.cpp_extension import load
owow = load(name="asd", sources=["/hdd/kng/CosineSampler/custom/grid_2d_original.cpp", "/hdd/kng/CosineSampler/custom/grid_2d_kernel_original.cu"])
# x, y -> t, x
class GridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, input, offset, point_num=None):
        print("1st F")
        # 0. preprocess
        grid = torch.cat([t, x], dim=-1).unsqueeze(0).unsqueeze(0)
        grid = grid.repeat([input.shape[0],1,1,1]) # repeat을 여기서 해주면 no grad 일테니까 repeat에 대한것을 나중에 반환때 Sum 해주어야 한다.
        
        N, C, IH, IW = input.shape  # 1 1 16 16
        _, H, W, _ = grid.shape # 1 1 2M 2   second 1 is always 1

        # grid = grid.to("cuda")   
        # x = x.to("cuda")
        # y = y.to("cuda")
        unnorm_it = grid[..., 0].contiguous()   # con()이 병목이려나..
        unnorm_ix = grid[..., 1].contiguous()

        input = input.view(N, C, IH*IW)

        # 1. normalize
        if offset:
            offset_tensor = torch.linspace(0,1-(1/N),N).to("cuda")
            it = owow.normalize_offset(unnorm_it, IH-2, offset_tensor)    
            ix = owow.normalize_offset(unnorm_ix, IW-2, offset_tensor)    
        else:
            it = owow.normalize(unnorm_it, IH-1)
            ix = owow.normalize(unnorm_ix, IW-1)

        # 2. compute corner indices
        with torch.no_grad():
            corner = owow.get_corner(it, ix)
            it_left = corner[0]
            it_right = corner[1]
            ix_top = corner[2]
            ix_bottom = corner[3]
       
        # 3. compute weights & get points
        weight = owow.get_weight(it, it_right, ix, ix_bottom)   # dx_right라는 중간 값이 필요함;;;
        dt_right = weight[0]
        dt_left = weight[1]
        dx_bottom = weight[2]
        dx_top = weight[3]



        point = owow.get_point(dt_right, dt_left, dx_bottom, dx_top)
        nw = point[0]
        ne = point[1]
        sw = point[2]
        se = point[3]

        # sanity checking
        with torch.no_grad():   # 나중에 쿠다로 바꾸기
            torch.clamp(it_left, 0, IW-1, out=it_left)
            torch.clamp(it_right, 0, IW-1, out=it_right)
            torch.clamp(ix_top, 0, IH-1, out=ix_top)
            torch.clamp(ix_bottom, 0, IH-1, out=ix_bottom)

        # 4. loop up values
        # with torch.no_grad():
        ix_top = ix_top.view(N,1,H*W)
        ix_bottom = ix_bottom.view(N,1,H*W) 
        it_right = it_right.view(N,1,H*W)
        it_left = it_left.view(N,1,H*W)

        vals = owow.gather(input, it_right, it_left, ix_bottom, ix_top, IW, W, H, C)
        nw_val = vals[0]
        ne_val = vals[1]
        sw_val = vals[2]
        se_val = vals[3]


        nw_val = nw_val.view(N, C, H, W)
        ne_val = ne_val.view(N, C, H, W)
        sw_val = sw_val.view(N, C, H, W)
        se_val = se_val.view(N, C, H, W)

        nw = nw.view(N, 1, H, W)
        ne = ne.view(N, 1, H, W)
        sw = sw.view(N, 1, H, W)
        se = se.view(N, 1, H, W)

        intpl = owow.interpolate(nw, nw_val, ne, ne_val, sw, sw_val, se, se_val, C)
        if offset:
            ctx.IH = IH - 2
            ctx.IW = IW - 2
        else:
            ctx.IH = IH - 1
            ctx.IW = IW - 1
            
        ctx.N = N
        ctx.C = C


        ctx.save_for_backward(t, x, it, ix, dt_right, dx_bottom, it_right, it_left, ix_bottom, ix_top,nw_val, ne_val, sw_val, se_val, nw, ne, sw, se, input)
        return intpl    # N C 1 E
    
    @staticmethod
    def backward(ctx, grad_out):
        print("1st B") 
        # CUDA
        N = ctx.N
        IH = ctx.IH
        IW = ctx.IW
        C = ctx.C
        # ix_right = ctx.ix_right  
        # iy_bottom = ctx.iy_bottom  # ctx랑 중복
        # nw_val = ctx.nw_val  
        # ne_val = ctx.ne_val  
        # sw_val = ctx.sw_val  
        # se_val = ctx.se_val  
        t, x, it, ix, dt_right, dx_bottom, it_right, it_left, ix_bottom, ix_top,nw_val, ne_val, sw_val, se_val, nw, ne, sw, se, input = ctx.saved_tensors
        return GridSampleBackward.apply(grad_out, t, x, input, it, ix, nw_val, ne_val, sw_val, se_val, dt_right, 
            dx_bottom, it_right, it_left, ix_bottom, ix_top, nw, ne, sw, se, N, C, IW, IH)  # B.f와 같음
        
        # 이거는 됨
        return GridSample_Backward.apply(grad_out, x,y,dy_top) 

class GridSampleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, t, x, input, it, ix, nw_val, ne_val, sw_val, se_val, dt_right,
        dx_bottom, it_right, it_left, ix_bottom, ix_top, nw, ne, sw, se, N, C, IW, IH): 
        print("2nd F")
        
        d_points = torch.stack([nw_val, ne_val, sw_val, se_val])
        ctx.N = N
        ctx.C = C
        ctx.IW = IW
        ctx.IH = IH
        d_grad = owow.interpolate_backward(grad_output, d_points, dt_right, 
            dx_bottom, it_right, ix_bottom, it, ix, N, C, IW, IH)
        cell_grad = owow.cell_backward(input, grad_output, it_right, it_left, ix_bottom, ix_top, nw, ne, sw, se, IW + 2,C)   
        # cell_grad = torch.zeros_like(input).flatten().scatter_(0,idx,val,reduce="add")
        # raise NotImplementedError
        # cell_grad = cell_grad.view(N, C, IW + 1, IH + 1)


        torch.set_printoptions(precision = 7)
        d_grad = d_grad.sum(1)

        cell_grad = cell_grad.view(N, C, IW+2, IH+2)
        ctx.save_for_backward(grad_output, d_points, dt_right, 
            dx_bottom, it_right, ix_bottom, it, ix, d_grad)
        
        return d_grad[2], d_grad[3], cell_grad, None, None
        # return d_grad[2], d_grad[3], None, None, None
       

    @staticmethod
    def backward(ctx, t_grad, x_grad, cell_grad, d, e):    # B.f에서 6개 리턴하니까 6개 받음
        print("2nd B")
        
        saved_grad_output, d_points, dt_right, dx_bottom, it_right, ix_bottom, it, ix, d_grad = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        IW = ctx.IW
        IH = ctx.IH
        d_grad = d_grad.squeeze(-1)
        t_grad = t_grad.squeeze(-1)
        x_grad = x_grad.squeeze(-1)

        bb_grad = owow.grad_backward_backward(d_grad[:2], t_grad, x_grad)
        dd_grad = owow.interpolate_backward_backward(saved_grad_output, d_points, t_grad, x_grad, dt_right, dx_bottom, 
            it_right, ix_bottom, it, ix, IW, IH)    # x_grad가 0이면 안곱하고 그런식으로 해야겠다 그리고 안곱한거는 zeros_like로 돌리면 됨
        print(dd_grad[0].sum(0).sum(0).shape)
        return bb_grad, dd_grad[0].sum(0).sum(0).squeeze(0).unsqueeze(-1), dd_grad[1].sum(0).sum(0).squeeze(0).unsqueeze(-1),  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None # B.f가 7개 받으니까 7개 리턴. grad_out은 None이고 n은 상수니까 None 리턴


# torch.manual_seed(12323)
torch.manual_seed(100)
n_cell=96
cell_dim = 4
point_num = 100000
W = 16
H = 16
cells = torch.rand(n_cell, cell_dim, W, H) 

# x = torch.linspace(0, 1, point_num)*2 - 1      # 2M
# y = torch.linspace(0, 1, point_num)*2 -1
# yx = np.random.rand(point_num, 2)
# yx[..., 1] = yx[..., 1] *2-1
# yx_f = torch.tensor(yx, requires_grad=True).float()
# t = yx_f[:, 0:1]
# x = yx_f[:, 1:2]

t0 = np.random.rand(point_num, 1)
x0 = np.random.rand(point_num, 1)

t = torch.tensor(t0, requires_grad= True).float()
x = torch.tensor(x0, requires_grad= True).float()

cells.requires_grad = True
# x.requires_grad = True
# t.requires_grad = True

cells = cells.to("cuda")
x = x.to("cuda")
t = t.to("cuda")
print("done")

grid_sample_2d = GridSample.apply
print("--forward--")
off = True
val2 = grid_sample_2d(t, x, cells, off, point_num)
# val2 = torch.tanh(val2)


val2 = val2.to("cpu")
print("---u2x---")
u2_x = torch.autograd.grad(
    val2, x, 
    grad_outputs=torch.ones_like(val2),
    retain_graph=True,
    create_graph=True
)[0]
print(u2_x.shape,'u2_x shape')
print("---u2t---")
u2_t = torch.autograd.grad(
    val2, t, 
    grad_outputs=torch.ones_like(val2),
    retain_graph=True,
    create_graph=True
)[0]
print(u2_t.shape,'u2_t shape')
print("---u2cell---")
u2_cell = torch.autograd.grad(
    val2, cells, 
    grad_outputs=torch.ones_like(val2),
    retain_graph=True,
    create_graph=True
)[0]
print(u2_cell.shape,'u2_cell shape')
print("---u2xx==--")
u2_xx = torch.autograd.grad(
    u2_x, x, 
    grad_outputs=torch.ones_like(u2_x),
    retain_graph=True,
    create_graph=True
)[0]
print(u2_xx.shape,'u2_xx shape')
print("--u2_yy--")
u2_tt = torch.autograd.grad(
    u2_t, t, 
    grad_outputs=torch.ones_like(u2_t),
    retain_graph=True,
    create_graph=True
)[0]
print(u2_tt.shape,'u2_tt shape')


# make_dot(val2, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val", format="png")
# make_dot(u2_x, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val_x", format="png")
# make_dot(u2_xx, show_attrs=True, show_saved=True).render("cpp_graphs/plz_tanh_tmp_val_xx", format="png")
print("----")


grid = torch.cat([t, x], dim=-1).unsqueeze(0).unsqueeze(0)
grid = grid.repeat([cells.shape[0],1,1,1])
val = grid_sample_temp.grid_sample_2d(cells, grid, step='cosine', offset=off)
# val = torch.tanh(val)
val = val.to("cpu")

u_cell = torch.autograd.grad(
    val, cells, 
    grad_outputs=torch.ones_like(val),
    retain_graph=True,
    create_graph=True
)[0]
print(u_cell.shape,'u_cell shape')

u_x = torch.autograd.grad(
    val, x, 
    grad_outputs=torch.ones_like(val),
    retain_graph=True,
    create_graph=True
)[0]
print(u_x.shape,'u_x shape')
u_xx = torch.autograd.grad(
    u_x, x, 
    grad_outputs=torch.ones_like(u_x),
    retain_graph=True,
    create_graph=True
)[0]
print(u_xx.shape,'u_xx shape')
u_t = torch.autograd.grad(
    val, t, 
    grad_outputs=torch.ones_like(val),
    retain_graph=True,
    create_graph=True
)[0]
print(u_t.shape,'u_t shape')
u_tt = torch.autograd.grad(
    u_t, t, 
    grad_outputs=torch.ones_like(u_t),
    retain_graph=True,
    create_graph=True
)[0]
print(u_tt.shape,'u_tt shape')

# u_cell = u_cell.flatten()
# u2_cell = u2_cell.flatten()
print('val == val2, max_error: {} at {}'.format((val.reshape(-1)-val2.reshape(-1)).abs().max(),(val.reshape(-1)-val2.reshape(-1)).abs().argmax()))
print('u_cell == u2_cell, max_error: {} at {}'.format((u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().max(),(u_cell.reshape(-1)-u2_cell.reshape(-1)).abs().argmax()))
print('u_x == u2_x, max_error: {} at {}'.format((u_x.reshape(-1)-u2_x.reshape(-1)).abs().max(),(u_x.reshape(-1)-u2_x.reshape(-1)).abs().argmax()))
print('u_y == u2_y, max_error: {} at {}'.format((u_t.reshape(-1)-u2_t.reshape(-1)).abs().max(),(u_t.reshape(-1)-u2_t.reshape(-1)).abs().argmax()))

print('u_xx == u2_xx, max_error: {} at {}'.format((u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().max(),(u_xx.reshape(-1)-u2_xx.reshape(-1)).abs().argmax()))
print('u_tt == u2_tt, mat_error: {} at {}'.format((u_tt.reshape(-1)-u2_tt.reshape(-1)).abs().max(),(u_tt.reshape(-1)-u2_tt.reshape(-1)).abs().argmax()))
# print('u_cell_x == u2_cell_x, max_error: {} at {}'.format((u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().max(),(u_cell_x.reshape(-1)-u2_cell_x.reshape(-1)).abs().argmax()))
# print('u_cell_t == u2_cell_y, max_error: {} at {}'.format((u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().max(),(u_cell_y.reshape(-1)-u2_cell_y.reshape(-1)).abs().argmax()))
# ''' ok before here'''
# print('u_x_cell == u2_x_cell, max_error: {} at {}'.format((u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().max(),(u_x_cell.reshape(-1)-u2_x_cell.reshape(-1)).abs().argmax()))
# print('u_y_cell == u2_y_cell, max_error: {} at {}'.format((u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().max(),(u_y_cell.reshape(-1)-u2_y_cell.reshape(-1)).abs().argmax()))
# # print('u_x_y == u2_x_y, max_error: {} at {}'.format((u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().max(),(u_x_y.reshape(-1)-u2_x_y.reshape(-1)).abs().argmax()))
# # print('u_y_x == u2_y_x, max_error: {} at {}'.format((u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().max(),(u_y_x.reshape(-1)-u2_y_x.reshape(-1)).abs().argmax()))

# # print('u_grid == u2_grid, max_error: {} at {}'.format(((u_grid.reshape(-1)-u2_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().max(),(u_grid.reshape(-1)-u2_grid.reshape(-1)).abs().argmax()))
# # print('u_grid_grid == u2_grid_grid, max_error: {} at {}'.format(((u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().max(),(u_grid_grid.reshape(-1)-u2_grid_grid.reshape(-1)).abs().argmax()))
# # print('u_input_grid == u2_input_grid, max_error: {} at {}'.format(((u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs()<1e-4).sum()==(number),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().max(),(u_input_grid.reshape(-1)-u2_input_grid.reshape(-1)).abs().argmax()))
# # print('u_grid_input == u2_grid_input, max_error: {} at {}'.format(((u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs()<1e-4).sum()==(number),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().max(),(u_grid_input.reshape(-1)-u2_grid_input.reshape(-1)).abs().argmax()))

# print('u_xx_cell == u2_xx_cell, max_error: {} at {}'.format((u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().max(),(u_xx_cell.reshape(-1)-u2_xx_cell.reshape(-1)).abs().argmax()))
# print('u_yy_cell == u2_yy_cell, max_error: {} at {}'.format((u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().max(),(u_yy_cell.reshape(-1)-u2_yy_cell.reshape(-1)).abs().argmax()))
    

# torch.set_printoptions(precision=10)
# print(u2_xx)
# print(u_xx)
# print(u2_yy)
# print(u_yy)
raise NotImplementedError



u2 = torch.tanh(val2)
print(val2)
print(u2)
# u2 = val2

# print(val2.shape)
# print(u2.shape)
# u2 =val2.squeeze()#.sum()
print("---u2x---")
u2_x = torch.autograd.grad(
    val2, x, 
    grad_outputs=torch.ones_like(val2),
    retain_graph=True,
    create_graph=True
)[0]
u2_x_tanh = torch.autograd.grad(
    u2, x, 
    grad_outputs=torch.ones_like(u2),
    retain_graph=True,
    create_graph=True
)[0]
print(u2_x)
print(u2_x_tanh)
# raise NotImplementedError
print("---u2x end---")

# print("---u2y---")
# u2_y = torch.autograd.grad(
#     val2, y, 
#     grad_outputs=torch.ones_like(val2),
#     retain_graph=True,
#     create_graph=True
# )[0]

