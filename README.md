# Machine Learning Directed Optimization of Biohybrid Fin Shapes
## Disease Biophysics Group
### Written by John Zimmerman
Updated 6/14/2021

## Description: 
### Code related to the paper: 
### _Bioinspired Design of a Tissue Engineered Ray with Machine Learning_

Source code for simulating biohybrid swimming performance and using machine learning to iteratively select improved fin geometries

Fluid dynamics simulations are based in part off of work done by Oliver Hennigh, which can be found at the following:
[Github Link](https://github.com/loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow)

### Main Inlcuded Files:

#### Enviroment Files

|Filename| Description|
|-|-|
|`TF_LBM environment.yml`   | File detailing the working conda environment|
|`CMAES_env.yml`            | File detailing the working conda environment for CMAES|


#### ML-DO - Simulating Swimmers

|Filename| Description|
|-|-|
|`Swim Search.ipynb`	 | Main ML-DO Search algorithm for simulating rays|
|`SwimTravelMovie.ipynb`	 | After Sim - turning velocity profiles into movies|
|`PostDatabaseSearch.ipynb` | Evaluating ML model, and database results evaluation|


#### Control Studies for Evaluating Search Functions

|Filename| Description|
|-|-|
|`NN Sampling Evaluation.ipynb`    | Evaluating different search functions on a control dataset|
|`CMA-ES Compare.ipynb`            | Evaluating CMAES on a control dataset |
|`Directed_Evolution_Compare.ipynb`| Evaluating DE on a control dataset |

---
### Computer Hardware
All simulations and machine learning were performed Dell Precision 3630 Tower, 
equipped with an Intel Core i7-8700k CPU (3.7 GHz), 16 GB RAM, and an NVIDIA 
GeForce GTX 1080. Software was written in Python (V3.6.5), with TensorFlow 
(V2.0.0) used as backend for GPU acceleration.




## Included Files
---
`TF_LBM environment.yml`

#### Description:
Conda file for setting up the internal developing environment (IDE).
Includes which packages are used for the rest of the files. Unless otherwise specified, this
is the default enviroment used throughout for the rest of the code


---
`CMAES_env.yml`           

#### Description: 
Conda file for setting up the internal developing environment (IDE).
Includes which packages are used for the rest of the files. Specifically used for evaluating
CMAES in comparison to other search strategies (see CMA-ES Compare.ipynb)


---
`Swim Search.ipynb`

#### Description:
Main Jupyter notebook for integrating the total set of tasks
in using ML-DO to search for efficient swimming in biohybrid fin geometries.
This include generating a master list to store simulations results for each
sDNA configurations. Then generating the initial search of which shapes to 
examine. Finally calls the main search algorithm, which creates a kinematic
mesh of each fin shape, generates the vertex normal, solves their velocities
for each frame, and then inserts this into the LBM fluid dynamics simulation.
LBM model than simulates the relative travel speed of each swimmer, which is
used as training data for the neural network model. Model then makes iterative
suggestions on which swimmers to simulate next.

#### Requires:
|Filename| Description|
|-|-|
|`DBG_LBM.py`     | Code for performing a LBM fluid dynamics simulation|
|`GeoSwimmer.py`  | Code for generating fin shapes from sDNA sequences|
|`is_point.py`    | Code for checking if a lattice point is contained in a prism|
|`SwimMesh.py`    | Code for solving fin kinematics and discretizing shapes|
|`SwimNN.py`      | Code for defining the neural network model and displaying results|
|`Contraction_profile2.txt` | Relative contractile radius data per solid frame|

#### Estimated Times on Specified Hardware:
- Initial Database Generation: 5-15 minutes
- Kinematic Simulation: ~2-5 minutes to solve material interactions/ ray
- Fluid Dynamics Simulation: ~10-25 minutes per ray for 25 second simulation
- Machine Learning Training Time: ~15 minutes per iteration

####  Outputs:
`results_updated.pkl`
Main data file. Pickle database file containing the measured distance travelled by each 
swimmer, as well as it’s updated model label as predicted by the neural network

`CurrentSwimlist.txt`
The list of which swimmer sDNA sequences will be simulated next in the current generation.
Contained as an external file outside of ram in case of memory leak/ simulation crash in 
the middle of generations.

`currentgen.txt`
Counter file with an integer value listing the current generation number for swimmers. 
Contained as an external file outside of ram in case of memory leak/ simulation crash 
in the middle of generations.

`{sDNA}_UV_kinematics.txt`
File listing the sum of the horizontal (U) and vertical (V) velocity profiles generated by 
kinematic contraction for a given sDNA sequence (wildcard denoted by *)

`{sDNA}.h5`
File containing the 3D fluid velocities fields for the final stroke of simulation (1 second 
at 1 hz stimulation). Stored as a [1,x,y,z,vel] array with [1,x,y,z,0] = U,  [1,x,y,z,1] = V,
 [1,x,y,z,2] = W. can be used to reconstruct the final velocity fields.

`CVVideo//{sDNA}_flow.avi`
Video containing a top down projection of the fluid dynamics simulation for a single fin 
(default recorded at 15 fps)

`SwimVelocities//{sDNA}_velocity.txt`
Text file containing a recording of the instantaneous velocity of a biohybrid fin geometry
(given for each fluid frame)


Total Run Time for 20 generations (1200 simulations): ~4 weeks

---
`SwimTravelMovie.ipynb`

#### Description:
Jupyter notebook file. Generates a video of a given fin geometry swimming. 
Requires that you have  previously run an individual LBM simulation of that swimmer to 
generate the  relative velocity profile.

#### Requires:
|Filename| Description|
|-|-|
|`GeoSwimmer.py`  | Code for generating fin shapes from sDNA sequences|
|`SwimMesh.py`   | Code for solving fin kinematics and discretizing shapes|
|`Contraction_profile2.txt` | data on the relative contractile radius as a function of time|

#### Outputs:
Series of .png files graphing the swimming distance travelled by a geometry

#### Estimated Times on Specified Hardware:
Generating a single video: 2-5 minutes

---
`PostDatabaseSearch.ipynb`

#### Description: 
Jupyter notebook file. Can be used to validate the final machine learning model,
graph the resulting functional landscape, and find 'Top', 'Middle' and 'Bottom' ranked sDNA
seqeuences. Assumes you have already run the swimsearch algorithim. 

#### Requires:
|Filename| Description|
|-|-|
|`GeoSwimmer.py`  | Code for generating fin shapes from sDNA sequences|
|`SwimNN.py`     | Code for defining the neural network model and displaying results|

#### Outputs:
Functional landscape graph, and a mean square error estiamte

#### Estimated Times on Specified Hardware:
Generating a single video: ~15 minutes to train final model



---
`NN Sampling Evaluation.ipynb`

#### Description:
Control study for examing different neural network architectures, sampling 
methods, and normalization approaches. Uses a a priori defined notional data set to evaluate
relative performance. Data set is multimodal, and has synergestic combinations (defined by 
GenFooDataFrame function).

#### Requires:
|Filename| Description|
|-|-|
|`GeoSwimmer.py`  | Code for generating fin shapes from sDNA sequences|


#### Outputs:
provides a text output file which details the number of correctly identified top performing
sequences. Additionally, provides graphs of the resulting functional space as mapped by the
neural network model

#### Estimated Times on Specified Hardware:
~15 minutes to generate results database (can be reused once generated first time)
~2 minutes to train a network each generation
~2 hours to evaluate a given search method

---
`CMA-ES Compare.ipynb `

#### Description:
Control study for examing covariance matrix adapted evolutionary strategies as a
search algorithim compared to ML-DO. Uses a a priori defined notional data set to evaluate
relative performance. Data set is multimodal, and has synergestic combinations (defined by 
GenFooDataFrame function).

#### Requires:
|Filename| Description|
|-|-|
|`GeoSwimmer.py`  | Code for generating fin shapes from sDNA sequences|

#### Outputs:
Provides a text output file which details the number of correctly identified top performing
sequences. Additionally, provides graphs of the sampled points projected into a 2D functional
space representation.

#### Estimated Times on Specified Hardware:
~15 minutes to generate results database (can be reused once generated first time)
~4 hours to run the entire search

---
`Directed_Evolution_Compare.ipynb`

#### Description:
Control study for examing directed evolution as a search algorithim compared to
ML-DO. Uses a a priori defined notional data set to evaluaterelative performance. Data set is
multimodal, and has synergestic combinations (defined by GenFooDataFrame function).

#### Requires:
|Filename| Description|
|-|-|
|`GeoSwimmer.py`  | Code for generating fin shapes from sDNA sequences|

#### Outputs:
Provides a text output file which details the number of correctly identified top performing
sequences. Additionally, provides graphs of the sampled points projected into a 2D functional
space representation.

#### Estimated Times on Specified Hardware:
~15 minutes to generate results database (can be reused once generated first time)
~5 minutes to evaluate the entire search method


---
## License

Copyright 2024 - Kevin Kit Parker

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.