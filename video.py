import cv2
import glob
import os
from datetime import datetime


def frames_to_video(fps, save_path, frames_path, min_index, max_index, alert):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (455, 256))
    imgs = glob.glob(frames_path + "/*.jpg")
    frames_num = len(imgs)
    for i in range(min_index, max_index):
        if os.path.isfile("%s/%d.jpg" % (frames_path, i)):
            frame = cv2.imread("%s/%d.jpg" % (frames_path, i))
            if i in alert:
                # print("alert!")
                frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
            videoWriter.write(frame)
    videoWriter.release()
    return


if __name__ == '__main__':
    err_list=[]
    with open('err.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            err_list.append(int(line.strip()))


    t1 = datetime.now()
    frames_to_video(22, "result.mp4", '../../driving_dataset/data', 20000, 30000, err_list)
    t2 = datetime.now()
    print("Time cost = ", (t2 - t1))
    print("SUCCEED !!!")
