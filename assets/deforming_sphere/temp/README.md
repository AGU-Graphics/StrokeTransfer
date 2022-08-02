## Directory Structures

temp directory includes the internal data or additional visualizations of the command line tools in the following (in pipeline order):

- regression
  - annotation_plot: includes optional annotation plot image.
  - gbuffers: intermediate image data.
  - features: include feature field visualization. 
  - canonical_sections: include canonical section visualization.
  - weights: include weight field visualization of orientation regression.
  - model_matrix: include the model matrix of the orientation regression.
- transfer
  - gbuffers: intermediate image data.
  - features: include feature field visualization. 
  - canonical_sections: include canonical section visualization.
  - weights: include weight field visualization of orientation transfer.
  - smoothing: internal matrix data (e.g. discrete Hodge Star matrix) are stored.
  - view_orientations: transferred orientations and filtered orientations are visualized.
- anchor_points
  - anchor_points_%d: anchor point visualization at each level.
- stroke
  - index: anchor point indices used for the start points of drawn strokes.
  - angular.json: stores angle offset data to generate random stroke orientations.

## Gallery

Some of internal data are visualized as images.

#### Annotation Plot

Optional annotation plot for visualization.

<img src="regression/annotation_plot/annotation_plot_000.png" alt="annotation plot" width="40%">

#### Features

Selected set of the feature fields are listed here:

|  I_d  |  I_s  | n_x | H | d_S |
| ---- | ---- | ---- | ---- | ---- |
|  ![Id](regression/features/I_d/I_d_000.png)  | ![Is](regression/features/I_s/I_s_000.png)   | ![N_x](regression/features/N_x/N_x_000.png)  | ![H](regression/features/H/H_000.png)  | ![d_S](regression/features/D_S/D_S_000.png)  |

#### Canonical Sections

Selected set of the canonical sections are listed here:

|  I_d_parallel  |  n_perp |  o_perp |
| ---- | ---- | ---- |
|  ![I_d_parallel](regression/canonical_sections/I_d_parallel/I_d_parallel_000.png)   | ![n_perp](regression/canonical_sections/n_perp/n_perp_000.png)  | ![o_perp](regression/canonical_sections/o_perp/o_perp_000.png)  |


#### Anchor Points

|  level 1  |  level 2 |  level 3 | level 4 |
| ---- | ---- | ---- | ---- |
|  ![level 1](anchor_points/anchor_points_1/anchor_points_1_001.png)   | ![level 2](anchor_points/anchor_points_2/anchor_points_2_001.png)   | ![level 3](anchor_points/anchor_points_3/anchor_points_3_001.png)   | ![level 4](anchor_points/anchor_points_4/anchor_points_4_001.png)   |