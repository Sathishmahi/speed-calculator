## Vehicle Speed Calculation and Tracking Project

### Overview

This project is designed to calculate and track the speed of vehicles in a video stream while also identifying and indicating vehicles that exceed a predefined speed threshold. It achieves this by leveraging computer vision techniques, object detection using the YOLOv8 pretrained model, and custom tracking logic.

### Main Features

1. **Vehicle Detection:** The project uses the YOLOv8 pretrained model to detect and identify vehicles in each frame of the video.

2. **Custom Tracking:** A custom tracking class is implemented to assign unique tracking IDs to vehicles across frames. The class uses Intersection over Union (IoU) calculations to determine if objects in different frames are the same.

3. **Speed Calculation:** Vehicle speeds are calculated by measuring the time it takes for a vehicle to traverse a predefined distance between two lines in the video frame. The conversion factor from pixels to meters is used to compute real-world speeds.

4. **Speed Threshold:** A speed threshold of 35 m/s (equivalent to 126 kilometers per hour) is defined as the limit for normal vehicle speed. Vehicles exceeding this threshold are marked as overspeeding.

5. **Statistics:** The system keeps track of the number of vehicles entering and exiting the specified region, allowing for traffic analysis.

### Assumptions

- The real-world distance between "Line 1" and "Line 2" is assumed to be 50 meters.
- The corresponding pixel distance between "Line 1" and "Line 2" in the video frame is approximately 390 pixels.
- The frame rate of the video is 30 frames per second.

### Usage

1. Input Video: Provide the input video (e.g., "input.mp4") for analysis.
2. Configuration: Adjust the speed threshold, camera calibration, and other parameters as needed.
3. Run the Code: Execute the Python script to analyze the video and obtain vehicle speed and tracking results.

### Results

The project generates detailed tracking information, including vehicle IDs, entry and exit counts, and overspeeding vehicle alerts. This data can be used for traffic monitoring and analysis.

### Dependencies

- Python
- OpenCV
- YOLOv8 pretrained model (for vehicle detection)
- Custom tracking class

### Future Improvements

This project can be further enhanced by implementing additional features, such as license plate recognition, vehicle type classification, and integration with real-time traffic management systems.


## Tech Stack

**Language:** Python

**Libraries to Use:** ultralytics,cv2,cvzone

**UI:** StreamLit


## Run Locally

Clone the project

```bash
git clone https://github.com/Sathishmahi/speed-calculator.git
```

create , activate conda env and install requirements   

```bash
 bash env.sh 
```
run streamlit app

### i create and run the project in [neurolab](https://neurolab.ineuron.ai/) not  local machine so i use local tunnel to test the project if run the project in locally just run 
```bash
streamlit run src/speed_calculator/app.py
```
### if neurolab or colab run 
```bash
bash run.sh
```

## Demo-video




https://github.com/Sathishmahi/speed-calculator/assets/88724458/b715335a-d2e6-49aa-8c0f-1d98afa7b6ab

