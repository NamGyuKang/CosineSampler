import torch
from cosine_sampler_3d import _cosine_3d

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

class CosineSampler3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode="zeros", align_corners=True, kernel='cosine', multicell = True):
        if multicell:
            ctx.offset  =torch.linspace(0, 1-(1/input.shape[0]), input.shape[0]).to('cuda')    
        else:
            ctx.offset = torch.zeros(input.shape[0]).to('cuda')   

        output = _cosine_3d.forward(input, grid, ctx.offset, padding_mode_enum(padding_mode), align_corners, kernel_enum(kernel), multicell)
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
        grad_input, grad_grid = _cosine_3d.backward(grad_out, input, grid, offset, padding_mode_enum(padding_mode),
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

        gInput, gGrid, ggOut = _cosine_3d.backward_backward(gOutInput, gOutGrid, input, grid, gOut, offset,
                                                                    padding_mode_enum(padding_mode), align_corners, input_requires_grad, kernel_enum(kernel), multicell)
        ctx.save_for_backward(input, grid, gOut, gOutGrid)
        
        return gInput, gGrid, ggOut

    @staticmethod
    def backward(ctx, gOutgInput, gOutgGrid, gOutggOut):
        align_corners = ctx.align_corners
        padding_mode = ctx.padding_mode
        input, grid, gOut, gOutGrid,  = ctx.saved_tensors 
        
        input_requires_grad = gOutgInput is not None and (gOutgInput != 0.).any().item()
        gInput, ggOut = _cosine_3d.backward_backward_backward(input, grid, gOut, gOutGrid, gOutgGrid.contiguous(), ctx.offset,
                                                                    padding_mode_enum(padding_mode), align_corners, input_requires_grad, kernel_enum(ctx.kernel), ctx.multicell) 

        b_input, _, _ = CosineSamplerBackwardBackward.apply(input, grid, gOutggOut.contiguous(), torch.ones_like(gOutgInput), gOutGrid, ctx.offset, padding_mode, align_corners, ctx.kernel, ctx.multicell)

        return gInput+b_input, None, ggOut, None, None, None, None, None, None, None
