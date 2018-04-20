# NumberPlateLocalization
A high recall Number Plate Detector System in Python and OpenCV

Using Image Processing techniques with the help of Numpy and OpenCV libraries in Python, **NumberPlateLocalization** is a high recall Number Plate Detector in images of cars.

## Usage:
For using NumberPlateLocalization through the Console, use the following command.


`python NumberPlateLocalization.py [options]`

## Available Options:
The available options are as follows:

1. #### Mode:
The Mode option is used to specify whether you want to write the output images of the application to a folder named *FinalOutput*. The default mode is `None`. To write output images to the HDD, use Write Mode by using `--write`.

2. #### Show Steps:
The Show Steps option is used to show the intermediate steps of the Detector, i.e. the output images of the preprocessing logic of the application. The default mode is `False`. To show the intermediate steps, use Show Steps Mode by setting the showsteps flag like so: `--showsteps`

3. #### All Images:
This is a default option of the Detector application, although it can be used explicitly for code clarity while direct usage in other codebases. Usage: `--all`.

4. #### Single Image:
The Single Image option is used to specify a single image for the Detector to work on. The required argument for this option is the Image Name(specified along with extension) immediately following the `--single` flag. There must be an image of the provided name in the *Dataset* folder. Example usage is like so: `--single FILENAME`

5. #### Multi Image:
The Multi Image option is used to specify a number of images from the Dataset on which the Detector will work. The required argument is a number following the `--multi` flag. For a valid number **_n_**, the Detector uses the first **_n_** images from the dataset. If the number specified exceeds the total images in the Dataset, the Detector defaults to the All Images option. Usage: `--multi NUMBER`

The options can be used in conjunction to create various combinations. Some examples below:

* Detecting the plates of the first 10 images, and writing the output images to HDD:

`python NumberPlateLocalization.py --multi 10 --write`

* Detecting the plate from an image named *_21.png_* and showing the intermediate steps, and writing the output to HDD:

`python NumberPlateLocalization.py --single 21.png --showsteps --write`

## Results:
The Detector works with fairly accurate results for most of the images in the dataset. Some results are as follows:

![Success Image 1](https://raw.githubusercontent.com/saketk21/NumberPlateLocalization/master/FinalOutput/1-detected.png)
![Success Image 2](https://raw.githubusercontent.com/saketk21/NumberPlateLocalization/master/FinalOutput/11-detected.png)
![Success Image 3](https://raw.githubusercontent.com/saketk21/NumberPlateLocalization/master/FinalOutput/12-detected.png)

However, the Detector fails for some images showing false positives or concave polygons over the Number Plate region.

![Failure Image 1](https://raw.githubusercontent.com/saketk21/NumberPlateLocalization/master/FinalOutput/9-detected.png)
![Failure Image 1](https://raw.githubusercontent.com/saketk21/NumberPlateLocalization/master/FinalOutput/18-detected.png)

The Detector results can be improved by using certain Machine Learning optimizations to choose the value of Epsilon for Polygon Approximation, applying a threshold for the Number of White Pixels per total pixels in the area of the detected contour etc.
