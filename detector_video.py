# from cfg import cfg
from detector_image import Detector
import cv2
import cfg
from utils import *
import torchvision

def detect_video(video_path, detector):

    transforms = torchvision.transforms.Compose([  # 归一化，Tensor处理
        torchvision.transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(video_path)     #读取视频

    fps = cap.get(cv2.CAP_PROP_FPS)       #每秒传输的帧数

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),       #视频大小
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        #创建视频流写入对象，VideoWriter_fourcc为视频编解码器

    vout = cv2.VideoWriter()        #写入视频文件对象
    vout.open('Video\out_video.mp4', fourcc, fps, size, True)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_c = frame.copy()      #拷贝一份原图
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))        #转化成Image对象
            W, H, scale, img = narrow_image(frame)     #缩小至416*416
            frame = transforms(img).unsqueeze(0)
            out_bbx = detector(frame,0.8,cfg.ANCHORS_GROUP)     #传入探索
            out_bbx = enlarge_box(W, H, scale, out_bbx)        #反算至原图
            for box in out_bbx:
                pred = int(box[5])
                x1 = int(box[1]-box[3]/2)
                y1 = int(box[2]-box[4]/2)
                x2 = int(box[1]+box[3]/2)
                y2 = int(box[2]+box[4]/2)
                if pred == 0:
                    text = "no hat"
                    cv2.rectangle(frame_c, (x1-1, y1-20) ,(x2+1,y1),(0, 250, 0),-1)       #信息框
                    cv2.putText(frame_c, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 1)      #字信息
                    cv2.rectangle(frame_c, (x1, y1), (x2, y2), (0, 250, 0), 2)         #实际框
                else:
                    text = "hat"
                    cv2.rectangle(frame_c, (x1 - 1, y1 - 20), (x2 + 1, y1), (0, 0, 250), -1)  # 信息框
                    cv2.putText(frame_c, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 1)  # 字信息
                    cv2.rectangle(frame_c, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 实际框

            cv2.imshow('img', frame_c)       #拼接起来
            vout.write(frame_c)          #逐帧写入
            # cv2.waitKey(int(1000 / fps))
            cv2.waitKey(1)
        else:
            break

    cap.release()     #停止捕获视频
    cv2.destroyAllWindows()    #关闭相应的显示窗口


if __name__ == '__main__':
    detector = Detector(r"model\YOLO_net.pth")

    video_path = r'Video\test_video.mp4'

    detect_video(video_path,detector)
