# Computer-Assisted-Surrounding-Mapping-for-Blind-People


- **Project Name**: Computer Assisted Surrounding Mapping for Blind People 
- **Short Name for Project**: SAMB
- **Team Members**: Ravindranath Sawane (@ravindranath-sawane), Hritik Debnath (@Deathreper20), Jayvardhan Ghloap (@jayvardhan), Harshal  Patil
- **Repository Link**: https://github.com/ravindranath-sawane/Computer-Assisted-Surrounding-Mapping-for-Blind-People


## 🔥 About this project

In the last decade, there has been a lot of deployment and development of devices capable of guiding and helping the blinds in mapping their surroundings both indoors and outdoors. But still making it available for the masses is still a very big challenge for us. This paper hence aims to solve this problem. The making of such devices and making them accessible to people with visual impairments(PVIs) becomes a predominant practice and essentially should be available for their locomotion. Giving access to such devices/models for people with visual impairments becomes an essential element, both for understanding the surroundings and helping PVIs for mobility assistance. These needs have now become even more complex and satisfying the needs of each user increasing more in terms of representation. Hence, providing assistive devices to PVIs helps them know their surroundings better and makes them somewhat independent. In this system, we are going to utilize this technology to help visually impaired people to access the latest technologies without a screen.

## Requirements
- OpenCV 3.4
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- IP Webcam

## Object Detection
*Steps to be followed*

- Download official [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put it on top floder of project.

- Run the follow command to convert darknet weight file to keras h5 file.
```
python yad2k.py cfg\yolo.cfg yolov3.weights data\yolo.h5
```

- Run follow command to show the demo. 
```
python setup.py
