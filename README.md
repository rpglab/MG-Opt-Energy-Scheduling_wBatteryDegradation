## Battery Degradation-based Microgrid Energy Scheduling

This program solves the microgrid optimal energy scheduling problem considering of a usage-based battery degradation neural network model


### File Description
* 'Case16.dat' is a sample microgrid datasheet including (wind turbine, solar farm, BESS).
* 'nn_class.py' represents the defination of neural network which is applied to tranined NNBD model, we need to call out this function when we apply the NNBD model in the optimization program.
*  'SCUC_Battery_updated_BDCmethod.py' is the main optimization program for the micorgrid scheduling with the NNBD model. (The program uploaded here may not be exactly the same with the paper)
* 'trained_nn_1d.pt' is the trained machine learning model data (NNBD model)


### Environment (Python packages)
* pandas
* numpy
* pyomo
* torch
* matplotlib
* itertools
* time
* os
* gurobi (solver)


### Main program 'SCUC_Battery_updated_BDCmethod.py'
* You can adjust the alpha value (defined in the paper) by change the ReducePercentage parameter.
* The CMDS Models can be selected by choose the function ESSReduceUsage, function ESSReduceUsage2 and parameter Pess_Max.
* Currenty, CMDS-BCL is applied, to switch the model to CMDS-PBCL, simply comment the function ESSReduceUsage & parameter ESSFeedBackLimit, then uncomment the  function ESSReduceUsage2 & parameter ESSFeedBackLimit1.
* For the CMDS-BRL, you can uncomment power function of parameterPess_Max which make the maximum power output reduced per iteration.

## Citation:
If you use these codes for your work, please cite the following paper:

Cunzhi Zhao and Xingpeng Li, “Microgrid Optimal Energy Scheduling Considering Neural Network based Battery Degradation”, *IEEE Transactions on Power Systems*, early access, Jan. 2023.


Paper website: <a class="off" href="/papers/CunzhiZhao-NNBD-MDS/"  target="_blank">https://rpglab.github.io/papers/CunzhiZhao-NNBD-MDS/</a>


## Contributions:
Cunzhi Zhao developed this program. Xingpeng Li supervised this work.


## Contact:
If you need any techinical support, please feel free to reach out to Cunzhi Zhao at czhao20@uh.edu.

For collaboration, please contact Dr. Xingpeng Li at xli83@central.uh.edu.

Website: https://rpglab.github.io/


## License:
This work is licensed under the terms of the <a class="off" href="https://creativecommons.org/licenses/by/4.0/"  target="_blank">Creative Commons Attribution 4.0 (CC BY 4.0) license.</a>


## Disclaimer:
The author doesn’t make any warranty for the accuracy, completeness, or usefulness of any information disclosed and the author assumes no liability or responsibility for any errors or omissions for the information (data/code/results etc) disclosed.
