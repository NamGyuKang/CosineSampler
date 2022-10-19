# CosineSampler


### __The reason for using both channel size and multi-grid__<br>
The cell representations will be the inputs to an MLP, and the channel size of cell representations is the input dimension of the MLP. It also has to do with the output dimension of PDE solutions (the higher the output dimensions of PDE solutions, the more input channels might be required). We empirically found the channel size showing good accuracy and computation tradeoffs.

### __Some readers with background from numerical mathematics / numerical linear algebra might relate the term multigrid to the multigrid method for solving linear systems arising from the discretization of PDEs.__<br>
We do really appreciate your comments. As you understood correctly, the step size (or the number of discretization points) is constant in the proposed method. Variable step size will be an exciting research direction we would like to explore in the near future. 

### __There should be a t^, instead of y^ in Equation (2), between line 64 and 65. The paper should clearly state the bounds of the spatial and temporal domains__<br>
If we get accepted, we will modify and add more detailed information.

### __Furthermore, it would be interesting to see experiments with decreasing number of channels c and number of grids M.__
