
class Tracking:
    def __init__(self, threshold=0.60):
        self.final_dict = {}
        self.count = 0
        self.threshold = threshold

    def track(self, box, frame_1=True):
        temp = 0
        if frame_1:
            self.final_dict[box] = self.count
            self.count += 1
            return self.count

        if self.final_dict:
            for pt, obj_id in self.final_dict.items():
                iou = self.calculate_iou(box, pt)
                if iou >= self.threshold:
                    del self.final_dict[pt]
                    self.final_dict[box] = obj_id
                    temp += 1
                    return obj_id
            if not temp:
                self.final_dict[box] = self.count
                self.count += 1
                return self.count


    @staticmethod
    def calculate_iou(box1, box2):
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