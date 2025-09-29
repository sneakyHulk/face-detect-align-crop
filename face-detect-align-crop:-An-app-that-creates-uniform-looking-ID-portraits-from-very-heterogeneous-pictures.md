---
title: "face-detect-align-crop: An app that creates uniform looking ID portraits from very heterogeneous pictures"
date: 2023-12-29T00:58:23Z
draft: false
websiteURL: https://heyn.dev
websiteName: heyn.dev
---

My university has a strong student community.
Large events are held on a regular basis.
To keep costs down, many students volunteer to help with various tasks, such as cooking or selling food and drinks.
This requires a certain amount of coordination and sometimes strict separation from the guests.
Portrait ID cards are used for this purpose.
To produce these, a photo of each volunteer is required.
However, the photos vary in size, quality and orientation.
This is exactly what this app solves: It unifies the submissions.

## How does it work?

This app uses five steps to generate the ID portraits.
The first step is to detect faces in the images.
There are three technologies to choose from for recognizing faces in images: Google's Mediapipe framework, Dlib's HOG descriptor and Dlib's CNN solution.
When a face is found, the coordinates of the corners of the bounding box around the face are returned.
Multiple results are returned if more than one face is found.
When using Mediapipe or CNN, the coordinates of both eyes are also returned.

With this data, we want to extract the faces:
To get the maximum use of pixels out of the original image and to ensure a certain uniformity, the image is expanded with an appropriate amount of white so that a crop or rotation does not fail and an image of the desired size can be created.
Then a translation of the original image is done such that the center of the expanded image matches the center of the face.
The location of the center is based on either the center of both eyes, if available, or the center of the bounding box.

Next, if the eye coordinates are available, the angle between them is computed.
With this angle, a rotation matrix is generated, which serves to rotate the translated image.

After that, the rotated image is scaled such that the output width or height corresponds to the width or height of the bounding box.
Which scaling factor is smaller determines whether the width or height should match.

In a final step, the image is cropped to the output width and height at the location of the face.

## Features of the GUI

The GUI allows you to streamline the process of generating portraits.
First, you can select multiple files or entire folders using a picker.
The image files are imported and displayed in a grid.
Now you may select images.
Clicking on an image will select it, clicking on it again will deselect it.
By clicking the Delete button, the selected imported images will be deleted.
By clicking the Start button while a selection is active, only the selected images will be processed, otherwise all images will be processed.

After that, the processing of multiple images is started in parallel via Python multiprocessing.
It will use Google's Mediapipe framework.
When an image is finished processing, a selection page is displayed.
This page is split in two, with the original image on the left and the generated portraits on the right.
In the original image, the bounding boxes of the found faces are highlighted in red.
On the right, you can select the best image.
Only one image can be selected at a time.
When you click Save, the selected image is saved for ID card creation and the next processed image with its portraits is displayed.
If no portrait is generated or the generated portrait is not satisfactory, other portrait generation solutions from Dlib can be clicked.
It will process the image in the background while continuing with the next image.
Attention: Do not spam click the CNN solution.
It will use up all your RAM and crash the application.

## Where can I get it?

The code is hosted on GitHub https://github.com/sneakyHulk/face-detect-align-crop.
It requires
- Tkinter,
- Pillow,
- mediapipe,
- dlib,
- opencv,
- and numpy

to run.