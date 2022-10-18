import torch
import time
import numpy as np

from torch.utils.cpp_extension import load
owow = load(name="asd", sources=["/hdd/kng/VOXEL/PIXEL_backup/utils/grid_2d.cpp", "/hdd/kng/VOXEL/PIXEL_backup/utils/grid_2d_kernel.cu"])

def grid_sampler_2d(input, grid, step = 'bilinear', offset = True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IH, IW = input.shape
    
   
    _, H, W, _ = grid.shape

    input = input.view(N, C, IH*IW)

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    # (iy,ix) will be the indices of the input
    # 1. normalize coordinates 0 to 1 (from -1 to 1)
    # 2. scaling to input size
    # 3. adding offset to make non-zero derivative interpolation
    unnorm_ix = grid[..., 0].contiguous()
    unnorm_iy = grid[..., 1].contiguous()
# 1. normalize
    if offset:
        offset_tensor = torch.linspace(0,1-(1/N),N).to("cuda")
        ix = owow.normalize_offset(unnorm_ix, IW-2, offset_tensor)    
        iy = owow.normalize_offset(unnorm_iy, IH-2, offset_tensor)    
    else:
        ix = owow.normalize(unnorm_ix, IW-1)
        iy = owow.normalize(unnorm_iy, IH-1)

    # 2. compute corner indices
    with torch.no_grad():
        corner = owow.get_corner(ix, iy)
        ix_left = corner[0]
        ix_right = corner[1]
        iy_top = corner[2]
        iy_bottom = corner[3]
    
    # # 3. compute weights & get points
    # weight = owow.get_weight(ix, ix_right, iy, iy_bottom)   # dx_right라는 중간 값이 필요함;;;
    # dx_right = weight[0]
    # dx_left = weight[1]
    # dy_bottom = weight[2]
    # dy_top = weight[3]

    
    # compute weights
    dx_right = step_f(ix_right-ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom-iy)
    dy_top = 1 - dy_bottom

    point = owow.get_point(dx_right, dx_left, dy_bottom, dy_top)
    nw = point[0]
    ne = point[1]
    sw = point[2]
    se = point[3]

    # sanity checking
    with torch.no_grad():   # 나중에 쿠다로 바꾸기
        torch.clamp(ix_left, 0, IW-1, out=ix_left)
        torch.clamp(ix_right, 0, IW-1, out=ix_right)
        torch.clamp(iy_top, 0, IH-1, out=iy_top)
        torch.clamp(iy_bottom, 0, IH-1, out=iy_bottom)

    # 4. loop up values
    # with torch.no_grad():
    iy_top = iy_top.view(N,1,H*W)
    iy_bottom = iy_bottom.view(N,1,H*W) 
    ix_right = ix_right.view(N,1,H*W)
    ix_left = ix_left.view(N,1,H*W)

    vals = owow.gather(input, ix_right, ix_left, iy_bottom, iy_top, IW, W, H, C)
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

    # intpl = owow.interpolate(nw, nw_val, ne, ne_val, sw, sw_val, se, se_val, C)


    # # for second differentiate
    
    row1 = torch.stack([(1-dy_bottom[0,0,:])*0.5*(IW-1), (1-dx_right[0,0,:])*0.5*(IH-1)],  -1)
    row2 = torch.stack([(dy_bottom[0,0,:])*0.5*(IW-1),  (dx_right[0,0,:]-1)*0.5*(IH-1)], -1)
    row3 = torch.stack([(dy_bottom[0,0,:]-1)*0.5*(IW-1),  (dx_right[0,0,:])*0.5*(IH-1)], -1)
    row4 = torch.stack([ (-dy_bottom[0,0,:])*0.5*(IW-1), (-dx_right[0,0,:])*0.5*(IH-1)], -1)
    w = torch.stack([row1, row2, row3, row4], 1)
    theta = torch.stack([se_val[0,0,0,:], ne_val[0,0,0,:], sw_val[0,0,0,:], nw_val[0,0,0,:]], -1)
    # print(w.shape)
    # w = w.repeat(1, C, 1, 1, 1)
    # w = w.view(w.shape[0], 4, 2)
    
    

    theta = theta.unsqueeze(1)
    
    re_second = torch.bmm(theta, w)
    re_second = re_second.view(W, 2)

    return re_second




def grid_sample_2d(input, grid, step='cosine', offset=True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    
    N, C, IH, IW = input.shape
    # grid = grid.repeat([N,1,1,1])
    # print(grid.shape)
    _, H, W, _ = grid.shape

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='smoothstep':
        step_f = lambda x: (x**2)*(3-2*x)
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    # (iy,ix) will be the indices of the input
    # 1. normalize coordinates 0 to 1 (from -1 to 1)
    # 2. scaling to input size
    # 3. adding offset to make non-zero derivative interpolation
    ix = grid[..., 0]
    iy = grid[..., 1]
     
    if offset:
        offset = torch.linspace(0,1-(1/N),N).reshape(N,1,1).to("cuda")
        # offset = torch.ones(iy.shape[0]).to("cuda")
        
        iy = ((iy+1)/2)*(IH-2) + offset; # -2 for extra cell by offset
        ix = ((ix+1)/2)*(IW-2) + offset;
    else:
        iy = ((iy+1)/2)*(IH-1)
        ix = ((ix+1)/2)*(IW-1)
    # if transform:
    #     theta = torch.randn(1).uniform_(0.2, 0.3)*torch.pi/180
    #     # print(theta)
    #     A = torch.tensor([[1, torch.tan(theta)],[0, 1]])
    #     # A = torch.tensor([[torch.cos(theta), torch.sin(theta)],[-torch.sin(theta), torch.cos(theta)]])
    #     xy = torch.stack([ix, iy], -1)[...,None]

    #     out_transfrom = torch.matmul(A, xy)
    #     Nt, Ct, Ht, Wt, _ = out_transfrom.shape
    #     out_transfrom =  out_transfrom.reshape(Nt, Ct, Ht, Wt)
        
    #     ix = out_transfrom[..., 0]
    #     iy = out_transfrom[..., 1]
                    
    

    # compute corner indices
    with torch.no_grad():
        ix_left = torch.floor(ix)
        ix_right = ix_left + 1
        iy_top = torch.floor(iy)
        iy_bottom = iy_top + 1
    
    # compute weights
    dx_right = step_f(ix_right-ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom-iy)
    dy_top = 1 - dy_bottom

    # dx_right = step_f(ix-ix_left)
    # dx_left = 1 - dx_right
    # dy_bottom = step_f(iy-iy_top)
    # dy_top = 1 - dy_bottom
    
    

    nw = dx_right*dy_bottom
    ne = dx_left*dy_bottom
    sw = dx_right*dy_top
    se = dx_left*dy_top
    
    

    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_left, 0, IW-1, out=ix_left)
        torch.clamp(ix_right, 0, IW-1, out=ix_right)
        torch.clamp(iy_top, 0, IH-1, out=iy_top)
        torch.clamp(iy_bottom, 0, IH-1, out=iy_bottom)


    input = input.view(N, C, IH*IW) # view
    # look up values
    nw_val = torch.gather(input, 2, (iy_top * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    ne_val = torch.gather(input, 2, (iy_top * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))
    sw_val = torch.gather(input, 2, (iy_bottom * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    se_val = torch.gather(input, 2, (iy_bottom * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))
    
    
    # temp = torch.pi*0.5*torch.sin(torch.pi * (ix_right- ix))
    # tempy = torch.pi*0.5*torch.sin(torch.pi * (iy_bottom- iy))
    # # # for second differentiate
    # row1 = torch.stack([(1-dy_bottom[0,0,:])*0.5*(IW-2)*temp, (1-dx_right[0,0,:])*0.5*(IH-2)*tempy],  -1)
    # row2 = torch.stack([(dy_bottom[0,0,:])*0.5*(IW-2)*temp,  (dx_right[0,0,:]-1)*0.5*(IH-2)*tempy], -1)
    # row3 = torch.stack([(dy_bottom[0,0,:]-1)*0.5*(IW-2)*temp,  (dx_right[0,0,:])*0.5*(IH-2)*tempy], -1)
    # row4 = torch.stack([ (-dy_bottom[0,0,:])*0.5*(IW-2)*temp, (-dx_right[0,0,:])*0.5*(IH-2)*tempy], -1)
    # w = torch.stack([row1, row2, row3, row4], -2).squeeze()
    # theta = torch.stack([se_val[0,0,:], ne_val[0,0,:], sw_val[0,0,:], nw_val[0,0,:]], -1)
    # # print(w.shape)
    # # w = w.repeat(1, C, 1, 1, 1)
    # # w = w.view(w.shape[0], 4, 2)
    
    # theta = theta.unsqueeze(1)
    
    # re_second = torch.bmm(theta, w)
    # re_second = re_second.view(W, 2)

    # temp2 = -(torch.pi**2)*0.5*torch.cos(torch.pi * (ix_right- ix))
    # tempy2 = -(torch.pi**2)*0.5*torch.cos(torch.pi * (iy_bottom- iy))
    # row1 = torch.stack([(1-dy_bottom[0,0,:])*0.5*(IW-2)*temp2*0.5*(IW-2), (1-dx_right[0,0,:])*0.5*(IH-2)*tempy2*0.5*(IH-2)],  -1)
    # row2 = torch.stack([(dy_bottom[0,0,:])*0.5*(IW-2)*temp2*0.5*(IW-2),  (dx_right[0,0,:]-1)*0.5*(IH-2)*tempy2*0.5*(IH-2)], -1)
    # row3 = torch.stack([(dy_bottom[0,0,:]-1)*0.5*(IW-2)*temp2*0.5*(IW-2),  (dx_right[0,0,:])*0.5*(IH-2)*tempy2*0.5*(IH-2)], -1)
    # row4 = torch.stack([ (-dy_bottom[0,0,:])*0.5*(IW-2)*temp2*0.5*(IW-2), (-dx_right[0,0,:])*0.5*(IH-2)*tempy2*0.5*(IH-2)], -1)
    # w = torch.stack([row1, row2, row3, row4], -2).squeeze()
    # theta = torch.stack([se_val[0,0,:], ne_val[0,0,:], sw_val[0,0,:], nw_val[0,0,:]], -1)
    

    # theta = theta.unsqueeze(1)
    
    # re_third = torch.bmm(theta, w)
    # re_third = re_third.view(W, 2)
   
    # w = torch.stack([-dx_left[0,0,:], dx_left[0,0,:], -dx_right[0,0,:], dx_right[0,0,:]], -1)
    # theta = torch.stack([se_val[0,0,:], ne_val[0,0,:], sw_val[0,0,:], nw_val[0,0,:]], -1)
    
    # re_second = torch.bmm(theta.view(theta.shape[0], 1, 4), w.view(w.shape[0], 4, 1)).view(-1, 1)
    
    # bilinear interpolation
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))
    
    return out_val 


def grid_sample_3d(input, grid, step='cosine', offset=False, end_modify=False):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IT, IH, IW = input.shape
    grid = grid.view(N, 1, grid.shape[2], grid.shape[-1])
    _, H, W, _ = grid.shape
    if step=='trilinear':
        step_f = lambda x: x
    elif step=='smoothstep':
        step_f = lambda x: (x**2)*(3-2*x)
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError
    # (iy,ix) will be the indices of the input
    # 1. normalize coordinates 0 to 1 (from -1 to 1)
    # 2. scaling to input size
    # 3. adding offset to make non-zero derivative interpolation
    it = grid[..., 0]
    ix = grid[..., 1]
    iy = grid[..., 2]
   
    if offset:
        offset = torch.linspace(0,(1-(1/(N))),N).reshape(N,1,1)
    else:
        offset = 0.0

    if end_modify:
        idx_t = torch.where(it == it[0,0,:].max())
        idx_x = torch.where(ix == ix[0,0,:].max())
        idx_y = torch.where(iy == iy[0,0,:].max())    
        it[idx_t[0], idx_t[1], idx_t[2]] -= 1e-7
        ix[idx_x[0], idx_x[1], idx_x[2]] -= 1e-7
        iy[idx_y[0], idx_y[1], idx_y[2]] -= 1e-7
        it = ((it+1)/2)*(IT-1) + offset
        ix = ((ix+1)/2)*(IW-1) + offset
        iy = ((iy+1)/2)*(IH-1) + offset
    else:
        if offset:
            it = ((it+1)/2)*(IT-2) + offset
            ix = ((ix+1)/2)*(IW-2) + offset
            iy = ((iy+1)/2)*(IH-2) + offset
        else:
            it = ((it+1)/2)*(IT-1)
            ix = ((ix+1)/2)*(IW-1)
            iy = ((iy+1)/2)*(IH-1)


    with torch.no_grad():
        it_nw_front = torch.floor(it)
        ix_nw_front = torch.floor(ix)
        iy_nw_front = torch.floor(iy)
        
        it_sw_front = it_nw_front
        ix_sw_front = ix_nw_front
        iy_sw_front = iy_nw_front+1

        it_ne_front = it_nw_front
        ix_ne_front = ix_nw_front+1
        iy_ne_front = iy_nw_front

        it_se_front = it_nw_front
        ix_se_front = ix_nw_front+1
        iy_se_front = iy_nw_front+1

        it_nw_back = it_nw_front+1
        ix_nw_back = ix_nw_front
        iy_nw_back = iy_nw_front

        it_ne_back = it_nw_front+1
        ix_ne_back = ix_nw_front+1
        iy_ne_back = iy_nw_front

        it_sw_back = it_nw_front+1
        ix_sw_back = ix_nw_front
        iy_sw_back = iy_nw_front+1

        it_se_back = it_nw_front+1
        ix_se_back = ix_nw_front+1
        iy_se_back = iy_nw_front+1

    # compute 3d weights
    step_it = step_f(it_se_back - it)
    step_ix = step_f(ix_se_back - ix)
    step_iy = step_f(iy_se_back - iy)
    nw_front = step_it * step_ix * step_iy
    ne_front = step_it * (1-step_ix) * (step_iy)
    sw_front = step_it * (step_ix) * (1-step_iy)
    se_front = step_it * (1-step_ix) * (1-step_iy)

    nw_back = (1-step_it) * step_ix * step_iy
    ne_back = (1-step_it) * (1-step_ix) * (step_iy)
    sw_back = (1-step_it) * (step_ix) * (1-step_iy)
    se_back = (1-step_it) * (1-step_ix) * (1-step_iy)
    
    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_nw_front, 0, IH-1, out=ix_nw_front)
        torch.clamp(iy_nw_front, 0, IW-1, out=iy_nw_front)
        torch.clamp(it_nw_front, 0, IT-1, out=it_nw_front)
        
        torch.clamp(ix_ne_front, 0, IH-1, out=ix_ne_front)
        torch.clamp(iy_ne_front, 0, IW-1, out=iy_ne_front)
        torch.clamp(it_ne_front, 0, IT-1, out=it_ne_front)
        
        torch.clamp(ix_sw_front, 0, IH-1, out=ix_sw_front)
        torch.clamp(iy_sw_front, 0, IW-1, out=iy_sw_front)
        torch.clamp(it_sw_front, 0, IT-1, out=it_sw_front)
        
        torch.clamp(ix_se_front, 0, IH-1, out=ix_se_front)
        torch.clamp(iy_se_front, 0, IW-1, out=iy_se_front)
        torch.clamp(it_se_front, 0, IT-1, out=it_se_front)

        torch.clamp(ix_nw_back, 0, IH-1, out=ix_nw_back)
        torch.clamp(iy_nw_back, 0, IW-1, out=iy_nw_back)
        torch.clamp(it_nw_back, 0, IT-1, out=it_nw_back)
        
        torch.clamp(ix_ne_back, 0, IH-1, out=ix_ne_back)
        torch.clamp(iy_ne_back, 0, IW-1, out=iy_ne_back)
        torch.clamp(it_ne_back, 0, IT-1, out=it_ne_back)
        
        torch.clamp(ix_sw_back, 0, IH-1, out=ix_sw_back)
        torch.clamp(iy_sw_back, 0, IW-1, out=iy_sw_back)
        torch.clamp(it_sw_back, 0, IT-1, out=it_sw_back)
        
        torch.clamp(ix_se_back, 0, IH-1, out=ix_se_back)
        torch.clamp(iy_se_back, 0, IW-1, out=iy_se_back)
        torch.clamp(it_se_back, 0, IT-1, out=it_se_back)
    
    
    input = input.view(N, C, IT*IH*IW)
    
    #correct
    nw_front_val = torch.gather(input, 2, (iy_nw_front * IH*IT + (ix_nw_front * IT + it_nw_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_front_val = torch.gather(input, 2, (iy_ne_front * IH*IT + (ix_ne_front * IT + it_ne_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_front_val = torch.gather(input, 2, (iy_sw_front * IH*IT + (ix_sw_front * IT + it_sw_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    se_front_val = torch.gather(input, 2, (iy_se_front * IH*IT + (ix_se_front * IT + it_se_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    nw_back_val = torch.gather(input, 2, (iy_nw_back * IH*IT + (ix_nw_back * IT + it_nw_back)).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_back_val = torch.gather(input, 2, (iy_ne_back * IH*IT + (ix_ne_back * IT + it_ne_back)).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_back_val = torch.gather(input, 2, (iy_sw_back * IH*IT + (ix_sw_back * IT + it_sw_back)).long().view(N, 1, H * W).repeat(1, C, 1))
    se_back_val = torch.gather(input, 2, (iy_se_back * IH*IT + (ix_se_back * IT + it_se_back)).long().view(N, 1, H * W).repeat(1, C, 1))

    # print((nw_front_val.view(N, C, H, W) * nw_front.view(N, 1, H, W)).shape)
    # print(nw_front_val.view(N, C, H, W) * nw_front.view(N, 1, H, W))
    # print(nw_front_val.view(N, C, H, W).shape, nw_front.view(N, 1, H, W).shape)
    
    # bilinear interpolation
    out_val = ((nw_front_val.view(N, C, H, W) * nw_front.view(N, 1, H, W)) + 
               (ne_front_val.view(N, C, H, W) * ne_front.view(N, 1, H, W)) +
               (sw_front_val.view(N, C, H, W) * sw_front.view(N, 1, H, W)) +
               (se_front_val.view(N, C, H, W) * se_front.view(N, 1, H, W)) + 
               (nw_back_val.view(N, C, H, W) * nw_back.view(N, 1, H, W)) + 
               (ne_back_val.view(N, C, H, W) * ne_back.view(N, 1, H, W)) +
               (sw_back_val.view(N, C, H, W) * sw_back.view(N, 1, H, W)) +
               (se_back_val.view(N, C, H, W) * se_back.view(N, 1, H, W))) 
    return out_val



def grid_sample_adaptive_2d(input, input_coord, grid, param, func, step='cosine', offset=False):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IH, IW = input.shape
    
    # print(grid.shape)
    _, H, W, _ = grid.shape

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    # (iy,ix) will be the indices of the input
    # 1. normalize coordinates 0 to 1 (from -1 to 1)
    # 2. scaling to input size
    # 3. adding offset to make non-zero derivative interpolation
    ix = grid[..., 0]
    iy = grid[..., 1]
    if offset:
        offset = torch.linspace(0,1-(1/N),N).reshape(N,1,1)
        iy = ((iy+1)/2)*(IH-2) + offset; # -2 for extra cell by offset
        ix = ((ix+1)/2)*(IW-2) + offset;
    else:
        iy = ((iy+1)/2)*(IH-1)
        ix = ((ix+1)/2)*(IW-1)

    # sanity checking
    with torch.no_grad():
        torch.clamp(input_coord, 0, IW-1, out=input_coord)

    # if transform:
    #     theta = torch.randn(1).uniform_(0.2, 0.3)*torch.pi/180
    #     # print(theta)
    #     A = torch.tensor([[1, torch.tan(theta)],[0, 1]])
    #     # A = torch.tensor([[torch.cos(theta), torch.sin(theta)],[-torch.sin(theta), torch.cos(theta)]])
    #     xy = torch.stack([ix, iy], -1)[...,None]

    #     out_transfrom = torch.matmul(A, xy)
    #     Nt, Ct, Ht, Wt, _ = out_transfrom.shape
    #     out_transfrom =  out_transfrom.reshape(Nt, Ct, Ht, Wt)
        
    #     ix = out_transfrom[..., 0]
    #     iy = out_transfrom[..., 1]
                    

    grid_coord = torch.stack([iy, ix], -1).unsqueeze(3)
    input_coord = input_coord.unsqueeze(2)
    # print(grid_coord.min(), grid_coord.max())
    # print('>>>',input_coord.min(), input_coord.max())
    # sigma = 80
    relu = torch.nn.ReLU()
    distance = torch.sqrt(torch.sum((torch.square(grid_coord - input_coord)), -1))
    # rbf = RBF_method(param, func)
    # rbf_distance = rbf.f(distance)
    # print(distance.min(), distance.max())
    # h = 1-rbf_distance
    h = relu(1 - torch.exp(distance)/param)
    h = h.repeat(1,C,1,1).view(N*C, h.shape[-2], h.shape[-1])
    input = input.view(N * C, -1, 1)
    # print(h.shape, input.shape)
    out_val = torch.bmm(h, input)
    out_val = out_val.view(N, C, out_val.shape[-2], out_val.shape[-1])
    
    

    return out_val


# # ['gaussian', 'sech'
# #  'MQ', 'IMQ', 'IQ', 'gaussian'                               - infinity smooth
# #  'matern'& name, 'polynomial' & order, 'thinplate' & order]  - piecewise smooth
class RBF_method():
    def __init__(self, parameter, function = 'gaussian'):
        self.c = parameter
        if function == 'gaussian':
            self.f = self.gaussian
        elif function =='sech':
            self.f = self.hyperbolic_secant
        elif function == 'MQ':
            self.f = self.multiquadratic
        elif function == 'IMQ':
            self.f = self.inverse_multiquadratic
        elif function ==  'IQ':
            self.f = self.invers_quadratic
        elif function == 'matern':
            self.f = self.matern
        elif function == 'polynomial':
            self.f = self.polynomial
        elif function == 'thinplate':
            self.f = self.thinplate
    # infinity smooth function
    def hyperbolic_secant(self, r,  numpy = False):
        if numpy == True :
            return 1/np.cosh(self.c * r)
        return 1/torch.cosh(self.c * r)
    # infinity smooth function
    def multiquadratic(self, r, numpy = False):
        if numpy == True :
            return np.sqrt((self.c**2 +r**2))
        return torch.sqrt((self.c**2 + r**2))
    # infinity smooth function
    def inverse_multiquadratic(self, r, numpy = False):
        if numpy == True :
            return 1/np.sqrt(1+(self.c*r)**2)
        return 1/torch.sqrt(1+(self.c*r)**2)
    #infinity smooth function
    def invers_quadratic(self, r, numpy = False):
        return 1/(self.c**2 + r**2)
    def gaussian(self, r, numpy = False):
        if numpy == True:
            return np.exp(-self.c * (r**2))
        return torch.exp(-self.c * (r **2))

    # piecewise smooth function
    def matern(self, r, name = 'C6', numpy = False):
        if numpy == True:
            if name == 'C2':
                return np.exp(-self.c*r)*(self.c*r+1)
            elif name == 'C4':
                return np.exp(-self.c*r)*((self.c*r)**2 + 3*self.c*r + 3)
            elif name == 'C6':
                return np.exp(-self.c*r)*((self.c*r)**3 + 6*(self.c*r)**2 + 15*self.c*r + 15)
        if name == 'C2':
            return torch.exp(-self.c*r)*(self.c*r+1)
        elif name == 'C4':
            return torch.exp(-self.c*r)*((self.c*r)**2 + 3*self.c*r + 3)
        elif name == 'C6':
                return torch.exp(-self.c*r)*((self.c*r)**3 + 6*(self.c*r)**2 + 15*self.c*r + 15)
    # piecewise smooth function
    def polynomial(self, r, order = 3, numpy = False): # order = odd value, {'3' : cubic, '5' : quintic}
        if numpy == True:
            return self.c * np.abs(r)**order
        return self.c * torch.abs(r)**order
    # piecewise smooth function
    def thinplate(self, r, order = 0, numpy = False): # order : even value
        if numpy == True:
            return self.polynomial(order, numpy)*np.log(np.abs(r))
        return self.polynomial(order, numpy)*torch.log(torch.abs(r))
    



    


# ### Radial basis Function list ###    
# # ['gaussian', 'sech'
# #  'MQ', 'IMQ', 'IQ', 'gaussian'                               - infinity smooth
# #  'matern'& name, 'polynomial' & order, 'thinplate' & order]  - piecewise smooth

# def grid_sample_2d_RBF(input, input_coord, grid, parameter, function = 'sech'):
#     rbf = RBF_method(parameter, function)
#     N, C, IH, IW = input.shape
#     _, H, W, _ = grid.shape
    
#     # (iy,ix) will be the indices of the input
#     # 1. normalize coordinates 0 to 1 (from -1 to 1)
#     # 2. scaling to input size
#     # 3. adding offset to make non-zero derivative interpolation
#     offset = torch.linspace(0,1-(1/N),N).reshape(N,1,1)
#     # grid = grid.repeat([N,1,1,1])

#     ix = grid[..., 0]
#     iy = grid[..., 1]
    
#     # idx_t = torch.where(it == 1)
#     # idx_x = torch.where(ix == 1)
#     # idx_y = torch.where(iy == 1)    

#     # it[idx_t[0], idx_t[1], idx_t[2]] -= 1e-7
#     # ix[idx_x[0], idx_x[1], idx_x[2]] -= 1e-7
#     # iy[idx_y[0], idx_y[1], idx_y[2]] -= 1e-7
    
#     ix = ((ix+1)/2)*(IW-1) + offset
#     iy = ((iy+1)/2)*(IH-1) + offset

#     # idx_t2 = torch.where(it > IT - 1)
#     # idx_x2 = torch.where(ix > IW - 1)
#     # idx_y2 = torch.where(iy > IH - 1)
    
    

#     # it[idx_t2[0], idx_t2[1], idx_t2[2]] -= offset[idx_t2[0],:,:].squeeze()
#     # ix[idx_x2[0], idx_x2[1], idx_x2[2]] -= offset[idx_x2[0],:,:].squeeze()
#     # iy[idx_y2[0], idx_y2[1], idx_y2[2]] -= offset[idx_y2[0],:,:].squeeze()

#     grid_coord = torch.stack([ix, iy], -1)

#     distance = []
#     for i in range(grid_coord.shape[-2]):
#         dist = torch.sum((grid_coord[:,:,i,:].unsqueeze(2) - input_coord)**2, -1)   # [96, 4, 1, 2] - [96, 4, 256, 2]
#         knn = torch.topk(dist, 4, largest=False)[0]
#         idx1 = knn[:,:,0]
#         idx2 = knn[:,:,1]
#         idx3 = knn[:,:,2]
#         idx4 = knn[:,:,3]
        
#         zero = torch.zeros_like(idx1)
        
#         print(torch.stack([torch.tensor([zero, idx2 - idx1, idx3 - idx1, idx4 - idx1]),
#                                 torch.tensor([idx1 - idx2, zero, idx3- idx2, idx4 - idx2]),
#                                 torch.tensor([idx1 - idx3, idx2 - idx3, zero, idx4 - idx3]),
#                                 torch.tensor([idx1 - idx4, idx2 - idx4, idx3- idx4, zero])], -1))

#         distance.append(torch.tensor([[zero, idx2 - idx1, idx3 - idx1, idx4 - idx1],
#                                 [idx1 - idx2, zero, idx3- idx2, idx4 - idx2],
#                                 [idx1 - idx3, idx2 - idx3, zero, idx4 - idx3],
#                                 [idx1 - idx4, idx2 - idx4, idx3- idx4, zero]]))
#         print(distance[0].shape)
#         exit(1)
#     out_knn = torch.stack(distance, 0)
#     print(out_knn.shape)
    
    
#     dis_ls = []
#     for i in range(grid_coord.shape[-2]):
#         idx1 = torch.sum((grid_coord[:,:,i,:].unsqueeze(2) - input_coord[:,:, out_knn[:,:,i, 0]])**2)
#         idx2 = torch.sum((grid_coord[:,:,i,:].unsqueeze(2) - input_coord[:,:, out_knn[:,:,i, 1]])**2)
#         idx3 = torch.sum((grid_coord[:,:,i,:].unsqueeze(2) - input_coord[:,:, out_knn[:,:,i, 2]])**2)
#         idx4 = torch.sum((grid_coord[:,:,i,:].unsqueeze(2) - input_coord[:,:, out_knn[:,:,i, 3]])**2)
        
#         distance = torch.tensor([0, idx2 - idx1, idx3 - idx1, idx4 - idx1],
#                                 [idx1 - idx2, 0, idx3- idx2, idx4 - idx2],
#                                 [idx1 - idx3, idx2 - idx3, 0, idx4 - idx3],
#                                 [idx1 - idx4, idx2 - idx4, idx3- idx4, 0])
#         dis_ls.append(distance)
#     out_distance = torch.stack(dis_ls, 0)
#     print(out_distance.shape)
#     exit(1)

#     distance = np.sqrt(distance)
#     distance = rbf.f(distance, numpy = True)
#     inv_distance = np.linalg.inv(distance)
#     inv_distance = torch.tensor(inv_distance, requires_grad=True).float()


#     with torch.no_grad():
#         torch.clamp(ix_nw_front, 0, IW-1, out=ix_nw_front)
#         torch.clamp(iy_nw_front, 0, IH-1, out=iy_nw_front)
#         torch.clamp(it_nw_front, 0, IT-1, out=it_nw_front)
        
#         torch.clamp(ix_ne_front, 0, IW-1, out=ix_ne_front)
#         torch.clamp(iy_ne_front, 0, IH-1, out=iy_ne_front)
#         torch.clamp(it_ne_front, 0, IT-1, out=it_ne_front)
        
#         torch.clamp(ix_sw_front, 0, IW-1, out=ix_sw_front)
#         torch.clamp(iy_sw_front, 0, IH-1, out=iy_sw_front)
#         torch.clamp(it_sw_front, 0, IT-1, out=it_sw_front)
        
#         torch.clamp(ix_se_front, 0, IW-1, out=ix_se_front)
#         torch.clamp(iy_se_front, 0, IH-1, out=iy_se_front)
#         torch.clamp(it_se_front, 0, IT-1, out=it_se_front)

#         torch.clamp(ix_nw_back, 0, IW-1, out=ix_nw_back)
#         torch.clamp(iy_nw_back, 0, IH-1, out=iy_nw_back)
#         torch.clamp(it_nw_back, 0, IT-1, out=it_nw_back)
        
#         torch.clamp(ix_ne_back, 0, IW-1, out=ix_ne_back)
#         torch.clamp(iy_ne_back, 0, IH-1, out=iy_ne_back)
#         torch.clamp(it_ne_back, 0, IT-1, out=it_ne_back)
        
#         torch.clamp(ix_sw_back, 0, IW-1, out=ix_sw_back)
#         torch.clamp(iy_sw_back, 0, IH-1, out=iy_sw_back)
#         torch.clamp(it_sw_back, 0, IT-1, out=it_sw_back)
        
#         torch.clamp(ix_se_back, 0, IW-1, out=ix_se_back)
#         torch.clamp(iy_se_back, 0, IH-1, out=iy_se_back)
#         torch.clamp(it_se_back, 0, IT-1, out=it_se_back)


#     r_sw_front = torch.sqrt((it_sw_front - it)**2 + (ix_sw_front - ix)**2 + (iy_sw_front - iy)**2)
#     r_se_front = torch.sqrt((it_se_front - it)**2 + (ix_se_front - ix)**2 + (iy_se_front - iy)**2)
#     r_nw_front = torch.sqrt((it_nw_front - it)**2 + (ix_nw_front - ix)**2 + (iy_nw_front - iy)**2)
#     r_ne_front = torch.sqrt((it_ne_front - it)**2 + (ix_ne_front - ix)**2 + (iy_ne_front - iy)**2)
    
#     r_sw_back = torch.sqrt((it_sw_back - it)**2 + (ix_sw_back - ix)**2 + (iy_sw_back - iy)**2)
#     r_se_back = torch.sqrt((it_se_back - it)**2 + (ix_se_back - ix)**2 + (iy_se_back - iy)**2)
#     r_nw_back = torch.sqrt((it_nw_back - it)**2 + (ix_nw_back - ix)**2 + (iy_nw_back - iy)**2)
#     r_ne_back = torch.sqrt((it_ne_back - it)**2 + (ix_ne_back - ix)**2 + (iy_ne_back - iy)**2)
    
    
#     ne_front = rbf.f(r_ne_front, numpy= False) 


#     se_front = rbf.f(r_se_front, numpy= False)  
#     nw_front = rbf.f(r_nw_front, numpy= False)
#     sw_front = rbf.f(r_sw_front, numpy= False)
#     ne_back =  rbf.f(r_ne_back, numpy= False)
#     se_back =  rbf.f(r_se_back, numpy= False) 
#     nw_back =  rbf.f(r_nw_back, numpy= False) 
#     sw_back =  rbf.f(r_sw_back, numpy= False)
    
    
#     # look up values
#     input = input.view(N, C, IT*IH*IW)
#     # print(C*IT*IH*IW)
#     # print((iy_nw_front * IH*IT + (ix_nw_front * IT + it_nw_front)).long().view(N, 1, H * W).repeat(1, C, 1).max())
#     # exit(1)
#     #correct
#     nw_front_val = torch.gather(input, 2, (iy_nw_front * IH*IT + (ix_nw_front * IT + it_nw_front)).long().view(N, 1, H * W).repeat(1, C, 1))
#     ne_front_val = torch.gather(input, 2, (iy_ne_front * IH*IT + (ix_ne_front * IT + it_ne_front)).long().view(N, 1, H * W).repeat(1, C, 1))
#     sw_front_val = torch.gather(input, 2, (iy_sw_front * IH*IT + (ix_sw_front * IT + it_sw_front)).long().view(N, 1, H * W).repeat(1, C, 1))
#     se_front_val = torch.gather(input, 2, (iy_se_front * IH*IT + (ix_se_front * IT + it_se_front)).long().view(N, 1, H * W).repeat(1, C, 1))
#     nw_back_val = torch.gather(input, 2, (iy_nw_back * IH*IT + (ix_nw_back * IT + it_nw_back)).long().view(N, 1, H * W).repeat(1, C, 1))
#     ne_back_val = torch.gather(input, 2, (iy_ne_back * IH*IT + (ix_ne_back * IT + it_ne_back)).long().view(N, 1, H * W).repeat(1, C, 1))
#     sw_back_val = torch.gather(input, 2, (iy_sw_back * IH*IT + (ix_sw_back * IT + it_sw_back)).long().view(N, 1, H * W).repeat(1, C, 1))
#     se_back_val = torch.gather(input, 2, (iy_se_back * IH*IT + (ix_se_back * IT + it_se_back)).long().view(N, 1, H * W).repeat(1, C, 1))

    


#     val_tensor = torch.stack([nw_front_val.view(N, C, H, W), ne_front_val.view(N, C, H, W),
#                               se_front_val.view(N, C, H, W), sw_front_val.view(N, C, H, W), 
#                               nw_back_val.view(N, C, H, W), ne_back_val.view(N, C, H, W),
#                               se_back_val.view(N, C, H, W), sw_back_val.view(N, C, H, W)], dim=0)

#     w_coeff = torch.matmul(inv_distance, val_tensor.view((N, distance.shape[0], C* H * W))) 
#     w_coeff = w_coeff.view(8, N, C, H, W)
    
    
#     out_val = (w_coeff[0].view(N, C, H, W) * nw_front.view(N, 1, H, W) + 
#             w_coeff[1].view(N, C, H, W) * ne_front.view(N, 1, H, W) +
#             w_coeff[2].view(N, C, H, W) * se_front.view(N, 1, H, W) +
#             w_coeff[3].view(N, C, H, W) * sw_front.view(N, 1, H, W) +
#             w_coeff[4].view(N, C, H, W) * nw_back.view(N, 1, H, W) + 
#             w_coeff[5].view(N, C, H, W) * ne_back.view(N, 1, H, W) +
#             w_coeff[6].view(N, C, H, W) * se_back.view(N, 1, H, W) +
#             w_coeff[7].view(N, C, H, W) * sw_back.view(N, 1, H, W))/((nw_front.view(N, 1, H, W) + 
#             ne_front.view(N, 1, H, W)+ se_front.view(N, 1, H, W) +sw_front.view(N, 1, H, W)+
#             nw_back.view(N, 1, H, W) + ne_back.view(N, 1, H, W)+ se_back.view(N, 1, H, W) +
#             sw_back.view(N, 1, H, W)))  
    
#     # print(w_coeff)
#     return out_val




