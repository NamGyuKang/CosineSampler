# CosineSampler


## __The reason for using both channel size and multi-grid__<br>
The cell representations will be the inputs to an MLP, and the channel size of cell representations is the input dimension of the MLP. It also has to do with the output dimension of PDE solutions (the higher the output dimensions of PDE solutions, the more input channels might be required). We empirically found the channel size showing good accuracy and computation tradeoffs.

## __Some readers with background from numerical mathematics / numerical linear algebra might relate the term multigrid to the multigrid method for solving linear systems arising from the discretization of PDEs.__<br>
We do really appreciate your comments. As you understood correctly, the step size (or the number of discretization points) is constant in the proposed method. Variable step size will be an exciting research direction we would like to explore in the near future. 

## __There should be a t^, instead of y^ in Equation (2), between line 64 and 65. The paper should clearly state the bounds of the spatial and temporal domains__<br>
If we get accepted, we will modify and add more detailed information.

## __Furthermore, it would be interesting to see experiments with decreasing number of channels c and number of grids M.__<br>

|M\c|1|2|3|4|5|6|7|8|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|4|8.70e-02|1.58e-02|7.24e-03|2.79e-02|1.92e-03|1.65e-03|1.53e-03|1.35e-03|
|8|8.47e-02|1.47e-02|3.06e-03|1.07e-03|8.26e-04|9.19e-04|9.94e-04|8.56e-04|
|16|8.73e-02|1.89e-02|2.20e-03|8.17e-04|8.06e-04|6.67e-04|7.96e-04|6.52e-04|
|32|8.30e-02|5.26e-02|1.90e-03|9.85e-04|9.55e-04|1.07e-03|7.11e-04|7.52e-04|
|48|8.32e-02|5.11e-02|2.63e-03|1.45e-03|7.32e-04|6.75e-04|1.24e-03|4.40e-04|
|64|8.53e-02|2.76e-02|1.85e-03|9.88e-04|8.53e-04|7.64e-04|5.71e-04|4.89e-04|
|80|8.43e-02|2.75e-02|2.58e-03|9.38e-04|8.27e-04|1.25e-03|9.42e-04|5.01e-04|
|96|8.41e-02|3.14e-02|1.92e-03|6.49e-04|6.72e-04|5.36e-04|7.74e-04|5.68e-04|
|112|2.50e-02|2.50e-02|1.81e-03|8.32e-04|2.63e-03|6.26e-04|5.51e-04|4.46e-04|

<br>
It is the best performance of the Burgers equation with different multigrid and channels until 200 iterations. As shown in the above table, channel sizes less than four performed poorly. More channels (>4) improved the performance, but the gain was marginal.  
