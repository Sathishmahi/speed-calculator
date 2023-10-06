class Tracking:
    def __init__(self, threshold=0.60):
        """
        Initialize the Tracking class.

        Parameters:
        - threshold (float): IoU threshold for object tracking.
        """
        self.final_dict = {}  # Dictionary to store tracked objects and their IDs
        self.count = 0  # Counter for assigning object IDs
        self.threshold = threshold  # IoU threshold for object matching

    def track(self, box, frame_1=True):
        """
        Track objects based on their bounding boxes.

        Parameters:
        - box (tuple): A tuple containing the coordinates of the bounding box (x1, y1, x2, y2).
        - frame_1 (bool): Indicates whether it's the first frame (default is True).

        Returns:
        - obj_id (int): ID of the tracked object.
        """
        temp = 0  # Temporary variable to track if an object is matched in the current frame

        if frame_1:
            self.final_dict[box] = self.count  # Assign a new ID to the object
            self.count += 1  # Increment the counter for the next object
            return self.count

        if self.final_dict:
            for pt, obj_id in self.final_dict.items():
                iou = self.calculate_iou(box, pt)  # Calculate IoU between the new box and tracked objects

                if iou >= self.threshold:  # If IoU is above the threshold, match the object
                    del self.final_dict[pt]  # Remove the old entry
                    self.final_dict[box] = obj_id  # Update the entry with the new box
                    temp += 1  # Set temp to indicate a match
                    return obj_id

            if not temp:  # If no match found, assign a new ID to the object
                self.final_dict[box] = self.count
                self.count += 1
                return self.count

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        - box1 (tuple): Coordinates of the first bounding box (x1, y1, x2, y2).
        - box2 (tuple): Coordinates of the second bounding box (x1, y1, x2, y2).

        Returns:
        - iou (float): IoU value between the two bounding boxes.
        """
        # Extract coordinates of the two boxes
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        # Calculate the coordinates of the intersection
        x1_inter = max(x1_box1, x1_box2)
        y1_inter = max(y1_box1, y1_box2)
        x2_inter = min(x2_box1, x2_box2)
        y2_inter = min(y2_box1, y2_box2)

        # Calculate the area of intersection
        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)
        area_inter = width_inter * height_inter

        # Calculate the area of each bounding box
        area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

        # Calculate IoU
        iou = area_inter / (area_box1 + area_box2 - area_inter)

        return iou
