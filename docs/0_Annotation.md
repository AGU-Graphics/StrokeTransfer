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

#### 1. Open an Exemplar Image

If you run the annotation tool from our command line tool interface, the exemplar image will be loaded on the initial state.

You can also use ```Open Image``` tool button to open your exemplar image with file dialog.

#### 2. Making Annotations

You can add annotations using the following inputs.

|  Step  | Inputs | 
| ---- | ---- | 
|  Add a new vertex for the current annotation. | Click  |
|  Add the last vertex for the current annotation and append it to the annotation list. | Double-Click  |
|  Change the width of the current annotation. | Width slider control.  |
|  Select an annotation for the next target. | Click on the annotation list |
|  Remove an annotation.| Click on the annotation list and then push "-" button.  |
|  Show/hide annotations  | Visible controllers for annotations and orientations are located on the tool buttons. |



#### 3. Export Annotations

You can simply push the save button in the tool buttons, and then the file dialog will be shown to ask the output file location. Finally, the output data will be saved as json data.