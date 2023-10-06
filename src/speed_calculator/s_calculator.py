from utils import read_yaml
from const import PARAMS_YAML_PATH,CONFIG_YAML_PATH
from tracker import Tracking
from ultralytics import YOLO
import numpy as np
import cvzone
import os
import cv2


class SpeedCalCulator:

    def __init__(self):

        self.params = read_yaml(PARAMS_YAML_PATH)["params"]
        self.config = read_yaml(CONFIG_YAML_PATH)

        self.tracking = Tracking(self.params["iou_ther"])


    def variables_initializer_and_params_initializer(self):

        self.frame_1 = True
        self.car_in = []
        self.car_out = []
        self.car_in_count = 0
        self.car_out_count = 0
        self.car_in_dict = {}
        self.car_out_dict = {}
        self.already_calculated_list = []
        self.speed_list = []

        self.l1_l2_m = self.params.get("l1_l2_m")
        self.line1_coor = self.params.get("line1_coor")
        self.line2_coor = self.params.get("line2_coor")
        self.conf_ther = self.params.get("conf_ther")
        self.wanted_class_ids = self.params.get("wanted_class_ids")
        self.distance_bet_l1_l2_in_px = self.params.get("distance_bet_l1_l2_in_px")
        self.model_name = self.params.get("model_name")
        self.one_px = self.l1_l2_m/self.distance_bet_l1_l2_in_px

    def model_initializer(self):

        self.model = YOLO(self.model_name)

    def process(self,frame,c):
        for p in [self.line1_coor,self.line2_coor]:
            x1,y1,x2,y2 = p
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
        results = self.model.predict(frame,verbose = False )[0].boxes.data
        for result in results:
            x1,y1,x2,y2,ther,cls = result
            x1,y1,x2,y2,cls = [int(i) for i in [x1,y1,x2,y2,cls]]
            if cls in self.wanted_class_ids:
                id = self.tracking.track((x1,y1,x2,y2),self.frame_1)
                center_point = int((x1+x2)/2),int((y1+y2)/2)

                if center_point[-1] < 150 and center_point[-1] > 120:
                    self.car_in.append(id)
                    # cv2.putText( frame,f"#entre car {id}",(x1,y1-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1 )
                    cvzone.putTextRect( frame,f"#entre car {id}",(x1,y1-20),offset=2)

                    if id not in self.car_in_dict and id not in self.already_calculated_list :
                        self.car_in_dict[id] = {"frame":c,"y_centre":center_point[-1]}
                        self.car_in_count += 1


                if center_point[-1] > 500:
                    self.car_out.append(id)
                    # cv2.putText( frame,f"#exit car {id}",(x1,y1-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1 )
                    cvzone.putTextRect( frame,f"#exit car {id}",(x1,y1-20),offset=2)
                    
                    if id not in self.car_out_dict and id not in self.already_calculated_list:
                        self.car_out_count += 1
                        self.car_out_dict[id] = {"frame":c,"y_centre":center_point[-1]}
                        if id in self.car_in_dict:
                            frame_count_sec = (self.car_out_dict[id]["frame"] - self.car_in_dict[id]["frame"])/self.fps
                            one_sec  = frame_count_sec/frame_count_sec
                            centre_point_diff_mtr = ((self.car_out_dict[id]["y_centre"] - self.car_in_dict[id]["y_centre"])*self.one_px)/frame_count_sec
                            cv2.putText( frame,f"#id {id} : {centre_point_diff_mtr:.1f}",(x1,y1-30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1 )
                            
                            self.already_calculated_list.append(id)

                self.frame_1 = False
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
                # cv2.putText( frame,f"#id {id}",(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1 )
                cvzone.putTextRect( frame,f"#id {id}",(x1,y1-10) ,offset=2)
        # cv2.putText( frame,f"#CAR IN COUNT  {self.car_in_count}",(50,50),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),2 )
        # cv2.putText( frame,f"#CAR OUT COUNT  {self.car_out_count}",(50,70),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),2 )

        cvzone.putTextRect( frame,f"#CAR IN COUNT  {self.car_in_count}",(50,50) ,offset=5)
        cvzone.putTextRect( frame,f"#CAR OUT COUNT  {self.car_out_count}",(50,70) ,offset=5)

        return frame


    def w_h_fps_fc_initializer(self,input_video_path:str):

        cap = cv2.VideoCapture(input_video_path)
        self.w,self.h,self.fc,self.fps = np.array([  cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
              cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)],dtype=np.int32)
        cap.release()

    def video_writer(self,output_video_path:str,input_video_path:str):
        
        self.w_h_fps_fc_initializer(input_video_path)
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path,fourcc,self.fps,(self.w,self.h))

        for c in range(self.fc):

            suc,frame = cap.read()
            out_frame = self.process(frame, c)

            writer.write(out_frame)

        writer.release()
        cap.release()


    def combine_all(self):
        self.variables_initializer_and_params_initializer()
        self.model_initializer()

        speed_calculator_con = self.config["speed_calculator"]
        artifact_con = self.config["artifact"]
        arti_root_dir = artifact_con["root_dir"]
        speed_calculator_root_dir = speed_calculator_con["root_dir"]
        speed_calculator_root_path = os.path.join(arti_root_dir,speed_calculator_root_dir)

        os.makedirs(speed_calculator_root_path,exist_ok=True)

        input_video_file_name = speed_calculator_con["input_video_file_name"]
        output_video_file_name = speed_calculator_con["output_video_file_name"]
        input_video_path = os.path.join(speed_calculator_root_path,input_video_file_name)
        output_video_path = os.path.join(speed_calculator_root_path,output_video_file_name)
        

        self.video_writer(output_video_path,input_video_path)





if __name__ == "__main__":
    sc = SpeedCalCulator()
    sc.combine_all()