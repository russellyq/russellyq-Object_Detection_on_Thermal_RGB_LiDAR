## Object Detection on Thermal Images, RGB Images, LiDAR

This repo aims to develop a object detector witha combination of multiple modalities.

Fisrtly, objects are detected from each individual sensor, RGB camera, Thermal camera and LiDAR with each object`s probability.

Next, objects detected from Thermal camera and LiDAR are projected into RGB images.

Then, Intersection over Union for each objects are calculated.

Next, hungarian algorithm is applied to output objects over certian IoU threshold.

<!--
这是一段被注释的文本。
它可以有多行，并且可以包含Markdown格式的文本。
![demo1](doc/fusion_smoke%2000_00_00-00_00_30.gif)
![demo2](doc/fusion_smoke1%2000_00_00-00_00_30.gif)
-->
