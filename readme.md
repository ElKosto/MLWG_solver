# Planar Waveguide with Multilayer Coating Solver


## To do list:

- separate data with DBSCAN clustering
- use classes

- The general idea is to have 3 kind of solvers - manual for geometry at agiven wavelength and for the complete. This is needed for the final wg solver which does it automatically. In this way. This could be usful to make this project use FEM solver with an optimizer.

- Detect all the fundamental modes related to ML coating

Dones
- done Debug the real ref index
- done Converge on the input parameters format (float for tests, lists? or nd arrays for the rest?)
- done The correct comparizon of with MOLINO will be asymmetric waveguide.
- done Implement Bracketing for the solution search

## Description
This Python project is designed for simulating the optical properties of waveguide covered with multilayer dielectric films. It calculates the reflection and transmission coefficients, and effective refractive indices, of light interacting with layered structures. The solver employs the Transfer Matrix Method (TMM) to analyze the optical behavior of these films at different wavelengths.

## Features
- Calculation of transfer matrices for individual layers and multilayer stacks.
- Support for TE polarization so far.
- Computation of reflection and transmission coefficients from multilayer dielectric films.
- Optimization routines to solve for the effective refractive index of waveguides with multilayer cladding.
- Visualization of the effective refractive index across a range of wavelengths.
- Interactive sliders to adjust waveguide and cladding layer parameters and visualize the impact on dispersion.

## Dependencies
- `numpy`: For numerical operations.
- `scipy`: Specifically `scipy.optimize.brentq` for root finding.
- `matplotlib`: For plotting and interactive visualizations.

## Installation
Ensure you have Python 3 installed along with the required dependencies.


## Usage
Run the script with Python 3

The script will display an interactive plot with sliders for adjusting the core and cladding parameters of the waveguide. You can observe the change in effective refractive index across the specified wavelength range.


### Simple code example

```python
import numpy as np
from refractiveindex import RefractiveIndexMaterial
from src.visualization.GUI import run_gui_simple



lambda_0 = np.linspace(0.5, 2., 500)  # Free-space wavelength in micro-meters 

# Waveguide parameters
n_core = 2.1 # core refractive index
n_sub = 1.7607843331457966 # substrate refractive index
n_clad =  [1.4820739955470303,2.1409150049633197]# cladding refractive index
w = 0.5 # Thicknesses of the waveguide core in micrometers
w_clad = [0.1,0.1] # Thicknesses of each layer in um

m = 0 # mode number

run_gui_simple(lambda_0, n_core, n_sub, n_clad, w, w_clad, m)
```

### Key Functions
- `TM_boundary`: Calculates the boundary transfer matrix for a single layer.
- `TM_propagation`: Calculates the propagation transfer matrix within a layer.
- `TM_multilayer`: Computes the overall transfer matrix for a multilayer stack.
- `compute_reflection_transmission`: Determines the reflection and transmission coefficients for multilayer films.
- `calc_dispersion_ML`: Calculates the effective refractive index for multilayer waveguide cladding.
- `calc_dispersion`: Calculates the effective refractive index for a simpler model.

### Interactive Elements
The script uses `matplotlib.widgets` to create interactive sliders and buttons. These UI elements allow you to dynamically adjust the dimensions and refractive indices of the waveguide and its cladding layers, and immediately see the impact on the plot of effective refractive index vs. wavelength.



## License
This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments
This work is based on the principles of optical physics and the Transfer Matrix Method (TMM) for analyzing multilayer optical systems.

## How to Contribute
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or a pull request on the project's repository.

