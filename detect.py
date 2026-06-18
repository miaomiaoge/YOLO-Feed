# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

"""
Run inference on images, videos, directories, streams, etc.
python detect.py --weights weights/yolov5s.pt --source 0  # 0 open PC video
Usage - sources:

     python detect.py --weights weights/yolov5s.pt --source 0    # 1 open external video

   
    $ python path/to/detect.py --weights yolov5s.pt --source 1              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path

from queue import Queue

import torch
import torch.backends.cudnn as cudnn

import cv2 
import datetime
import serial
from time import sleep
import serial.tools.list_ports
import csv

import time
import threading

Qd=Queue()
Qt=Queue()
FLAG=True

def recv(ser):
    while True:
        if FLAG==False:
            return 
       
        data = ser.readline().decode()
        if data == '':
            continue
        else:
            break
        sleep(0.001)
   
    return data   #,TIME

def write_csv(csv_path,data_row):
    with open(csv_path,'a+') as f:
        csv_write = csv.writer(f)
        if not os.path.getsize(csv_path):
            print("====================")
            print("Creating new excel!")
            print("====================")
            cvs_head=["ID","Date"]
            csv_write.writerow(cvs_head)
        csv_write.writerow(data_row)
        
def Cap_tag(path):
    
    plist = list(serial.tools.list_ports.comports())
    print("plist:",plist)
    serialName = 'COM3'
    if len(plist) <= 0:
        print ("The Serial port can't find!")
    else:
        plist_0 =list(plist[1])
        serialName = plist_0[0]
        print ("check which port was really used >",serialName)
    
    ser = serial.Serial(serialName, 9600, timeout=0.5)
    if ser.isOpen():
        print("serial open success")
    else:
        print("serial open failed")
    global Qd,Qt,FLAG
    while True:
        
        data = recv(ser)
        if data is None:
            return 
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
  
        data_row =[data[1:19],date]
        Qd.put(data[1:19])
        Qt.put(date)
        #write_csv(path,data_row)



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=0.1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    
    # Load model
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    
    #############################################################################
    
    DaTa=''
    pre_data=''
    pre_time=''
    init_feed_count=-100
    cur_feed_count=-1
    new_flag=False
    
    if source=='0' or source=='1':
        DaTe='2017-07-17 06:03:00'
    else:
        DaTe=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    #############################################################################
    #DaTe=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    ################################# save raw video  ###########################
    
    save_p = time.strftime('%Y-%m-%d %H', time.localtime()).replace(" ","").replace("-","")+".mp4"  # force *.mp4 suffix on results videos
    vid_wr= cv2.VideoWriter(save_p, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    
    ################################ save raw video  ############################        
    
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        Flag=False

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
          
            p = Path(p)  # to Path
            
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            csv_path=str(Path(save_path).with_suffix('.csv')) 
            ############################################  save raw video  ###############################################
            
            if vid_cap:  # video
                
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
            vid_wr.write(im0)
            
            ############################################  save raw video  ###############################################  
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            txt = " feed_count : %d "%len(det)
  
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
   
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        label=None
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            ##############################################################
            ######################  new code begin  ######################                             
            ##############################################################
            if DaTa!='':                                         #     DaTa=''   ''    '001'    '001'     '001'     '002'     '003'
                pre_data=DaTa                                    # pre_data=|    |     '001'    '001'     '001'     '002'     '003'
                pre_time=DaTe	                                  # pre_time=|    |      1s       1s        3s        4s        5s
                 	
            global Qd,Qt                                         #              1st    2nd       3th       4th      5th       
            if not Qd.empty():                                   #    if    F    T       T        T        T         T         F
              
                if DaTa=='':                                     #    if    |    T       F        F        F         F         |
                    init_feed_count=len(det)                    # init_feed=|    10      |        |        |         |         |
                datet = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                datet.replace(" ","").replace("-","")
                interval=(datetime.datetime.strptime(datet,"%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime(DaTe,"%Y-%m-%d %H:%M:%S")).seconds>7
                                                                 # interval=|    T       F        T        F         F         |
                DaTa=Qd.get()                                    #     DaTa=|   '001'   '001'    '001'    '002'     '003'      |
                DaTe=Qt.get()                                    #     DaTe=|    1s      2s       3s       4s        5s        |
                if DaTa==pre_data:                               #    if    |    F       T        T        F         F
                    if interval:                                 #    if    |    |       F        T        |         |         |
                        new_flag=True                            # new_flag=|    |       |        T        |         |         |
                    else:
                        DaTe=pre_time                            #     DaTe=|    |       1s       |        |         |
                if pre_data=='':                                 
                    pre_data=DaTa                                # pre_data=|   '001'    |        |        |         |         |
                
            datet = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            datet.replace(" ","").replace("-","")
            Eq=(datetime.datetime.strptime(datet,"%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime(DaTe,"%Y-%m-%d %H:%M:%S")).seconds<10
                            
            if Eq and DaTa==pre_data and not new_flag:          #          F&T&T T&T&T  T&T&T   T&T&F    T&F&T     T&F&T       F&T&T
                Flag=True 
            else:
                if DaTa!="":                                    #      DaTa=''    |      |      '001'    '002'     '003'       '003'         
                    cur_feed_count=len(det)                     #  cur_feed=|     |      |        8        6        3           0
                    in_feed=init_feed_count-cur_feed_count      #   in_feed=|     |      |        2        2        3           3
                    #in_feed=max(in_feed,0)
                    print("{} take in feed {}.".format(DaTa,int(in_feed)))
                    with open(csv_path,'a+') as f:
                        csv_write = csv.writer(f)
                        if not os.path.getsize(csv_path):
                            print("====================")
                            print("Creating new excel!")
                            print("====================")
                            cvs_head=["ID","Date","init_feed","cur_feed","In_feed"]
                            csv_write.writerow(cvs_head)
                        data_row=[pre_data,pre_time,init_feed_count,cur_feed_count,in_feed]
                                                               #            |     |      |  ['01',1s,2]['01',3s,2]['02',4s,3]['03',5s,3]
                        csv_write.writerow(data_row)
                    if not Eq:                                 #       if   |     |      |        F        F        F          T
                        DaTa=''                                #       DaTa=|     |      |        |        |        |          ''
                    pre_data=''                                #   pre_data=|     |      |       ''        ''       ''         ''
                    pre_time=''                                #   pre_time=|     |      |       ''        ''       ''         ''
                    init_feed_count=len(det)                   #  init_feed=|     |      |        8        6        3          0
                    
            new_flag=False                                     #   new_flag=F     F      F        F        F        F          F
                          
            im0= cv2.putText(im0,DaTa,(450,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            im0= cv2.putText(im0, datet, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(im0,txt, (270, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
            
            ##############################################################
            ########################  new code end  ######################
            ##############################################################

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:

                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        
                        
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    ################################# save raw video  ###########################
    
    vid_wr.release() 
    
    ################################# save raw video  ########################### 


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    global FLAG
    FLAG=False

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    csv_name=time.strftime('%Y-%m-%d %H', time.localtime()).replace(" ","").replace("-","")+".csv"
    t1=threading.Thread(target=main,args=(opt,))
    t2=threading.Thread(target=Cap_tag,args=(os.path.join("CSV_Result",csv_name),))
    
    #启动线程
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
