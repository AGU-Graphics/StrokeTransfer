# Annotation Tool Manual

## User Interface

The tool consists of:
- Tool buttons: contains tool action for opening image, visible controls for annotations and orientations, data io functions for annotations.
- Main canvas: space where you can annotate orientations, and width (length will be automatically computed).
- Width slider: simple width controller for the next/selected target annotation.
- Color settings: contains key visual settings to change colors.
- Annotation list: simple list selection used for changing width or removing an annotation.

![annotation tool](../images/annotation/annotation_tool.png)


## Annotation Steps

#### 1. Open Exemplar Image

| Inputs   | Description | 
| ---- | ---- | 
|  Open Image Button (Ctrl + I) <br><img src="../python/util/tool/screen_shots/open_image_button.png"  height="64" alt="open image button"> | Open exemplar image (.png) with file dialog. |

Note: If you run the annotation tool from our command line tool interface, the exemplar image will be loaded on the initial state.


#### 2. Making Annotations

You can add annotations using the following inputs.

##### Insert Annotations
| Inputs   | Description | 
| ---- | ---- | 
|  Click | Add a new vertex for the current annotation.  |
| Double-Click | Add the last vertex for the current annotation and append it to the annotation list.   |
| Add Button (Enter) <br> <img src="../python/util/tool/screen_shots/add_button.png" height="32" alt="add button"> | Append the current annotation to the annotation list.   |

##### Select Annotation

| Inputs   |  Description |
| ---- | ---- |  
|  Select on Annotation List UI<br><img src="../python/util/tool/screen_shots/annotation_list.png" width="128" alt="width slider">   | Select an annotation for width control and deleting. |

##### Change Annotation Width

| Inputs   | Description | 
| ---- | ---- | 
| Width Slider <br><img src="../python/util/tool/screen_shots/width_slider.png" height="32" alt="width slider">  | Change the width of the selected annotation. <br><br> The width value is also used for current annotation.|

##### Delete Annotation

| Inputs   | Description | 
| ---- | ---- | 
| Delete Button (Delete)  <br> <img src="../python/util/tool/screen_shots/delete_button.png" height="32" alt="add button"> | 1. Select an annotation from Annotation List UI. <br> 2. Delete the selected annotation. |

##### Visible Control
| Inputs   | Description | 
| ---- | ---- | 
| Visible Annotation Button <br> <img src="../python/util/tool/screen_shots/visible_annotation_button.png" height="64" alt="add button"> |  Show/hide annotations.  |
| Visible Orientation Button <br> <img src="../python/util/tool/screen_shots/visible_orientation_button.png" height="64" alt="add button"> |  Show/hide interpolated orientations (vector field).  |



#### 3. Save Annotation Data

| Inputs   | Description | 
| ---- | ---- | 
| Save Annotation Button (Ctrl + S)  <br> <img src="../python/util/tool/screen_shots/save_annotation_button.png" height="64" alt="add button"> |Save annotation data (.json) with the file dialog. |

#### 4. Open Annotation Data

| Inputs   | Description | 
| ---- | ---- | 
| Open Annotation Button (Ctrl + O)  <br> <img src="../python/util/tool/screen_shots/open_annotation_button.png" height="64" alt="add button"> |Open annotation data (.json) with the file dialog. |