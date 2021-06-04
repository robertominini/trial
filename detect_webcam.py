from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import keyboard
import torch
from torchvision.ops import nms



def predict_faces_and_masks(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(height, width) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# setting images as blobs and calculating face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# setting up empty images where we will store detection data
	faces = []   #here we store the image data of the detected faces
	coords = []  #here we store the coordinates of the detected faces
	preds = []   #here we store the mask detection preictions

	# loop over the detections
	for detection in detections[0,0]:
		# extracting the confidence of the prediction
        
		confidence = detection[2]  

		# now we only consider dections with a confidence greater than 50%
        
		if confidence > 0.5:
			# we calculate the coordinates of the bounding boxes
			box = detection[3:7] * np.array([width, height, width, height])
			(x1, y1, x2, y2) = box.astype("int")
			(x1, y1) = (max(0, x1), max(0, y1))
			(x2, y2) = (min(width - 1, x2), min(height - 1, y2))

			# here we preprocess the faces
            
			face = frame[y1:y2, x1:x2]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)


			faces.append(face)
			coords.append((x1, y1, x2, y2))

	# if there is any face, we proceed with the mask prediction
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

    # here we return coords in order to draw the rectangles, preds in order to know whether the rectangles are red (no mask)
    # or green (mask detected) and we return len(faces) since this information will be used to support people counting.
	return (coords, preds, len(faces))



def draw_rectangles(frame, coords, preds):
    
    for (box, pred) in zip(coords, preds):  

        (x1, y1, x2, y2) = box
        (mask, withoutMask) = pred

		# here we constrct the appropriate label according to the mask/no_mask prediction
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# here we create a label with the confidence of the prediction
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# here we draw the actual rectangles
        cv2.putText(frame, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)









# here we load the RESNET model for FACE DETECTION

resnet_prototxt_PATH = r"resnet_face_detector\deploy.prototxt"
resnet_weights_PATH = r"resnet_face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(resnet_prototxt_PATH, resnet_weights_PATH)

# here we load the MOBILENET V2 model which we trained with the train_mask_detector.py file

model_name = "expanded_dataset_mask_detector"
maskNet = load_model(model_name)
print(f"<INFO> Loading model {model_name}.")

# here we load and prepare the YOLO v3 model for PEOPLE COUNTING

classes = None
with open('yolov3/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

yolov3_weights_PATH = r"yolov3\yolov3.weights"
yolov3_cfg_PATH = r"yolov3\yolov3.cfg"
net = cv2.dnn.readNet(yolov3_weights_PATH, yolov3_cfg_PATH)
    


# initialize the video stream
print("\n<INFO> Starting Live video stream. To end the video stream, please click on the window with the webcam capture and press B.")

vs = VideoStream(src=0).start()


        
# here we use an infinte while loop since the program needs to loop as long as the user desires. The loop can be 
# interrupted by pressing the key "B"

while True:
	# here we grab the frames and resize them to 500
    frame = vs.read()
    if frame is None:
        raise Exception("The frame from the webcam is empty. Check that your webcam is connected and fucntioning")
    frame = imutils.resize(frame, width = 500) 

	# with this function we detect faces in the frame and determine if they are wearing a
	# face mask or not. We also extract the number of faces
    (coords, preds, n_faces) = predict_faces_and_masks(frame, faceNet, maskNet)
    
    
    # setting up YOLOv3 model and making predictions
    
    Width = frame.shape[1]
    Height = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 0.004, (416,416), (0,0,0), True, crop=False))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    
    
    class_ids = []
    confidences = []
    boxes = []

    people = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:

                people += 1
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)

                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    
    ppl_boxes =[]
    ppl_confidences = []
    final_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i]==0:
            ppl_boxes.append(box)
            ppl_confidences.append(confidences[i])

    tens_ppl_boxes = torch.tensor(ppl_boxes, dtype=torch.float32)
    tens_scores = torch.tensor(ppl_confidences, dtype=torch.float32)

    try:
        final_indices = nms(boxes = tens_ppl_boxes, scores = tens_scores, iou_threshold=0.2)

    
        for i in final_indices:
            final_boxes.append(ppl_boxes[i])
            
        for i in indices:
            i = i[0]
            box = boxes[i]
            if class_ids[i]==0 and box in final_boxes:
                ppl_boxes.append(box)
                label = str(classes[class_id]) 
                #Here we write the rectangles for people
                cv2.rectangle(frame, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
                cv2.putText(frame, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
    except:
        None
    
    #with this function we write the rectangles for the faces
    
    draw_rectangles(frame,coords, preds)
    
    #here we add text to display the number of people and faces
    
    font                   = cv2.FONT_HERSHEY_COMPLEX
    bottomLeftCornerOfText = (10,300)
    fontScale              = 0.7
    fontColor              = (0,0,0)
    lineType               = 2
    
    cv2.putText(frame,f'Number of people: {max(len(final_boxes), n_faces)} (ppl: {len(final_boxes)}, faces: {n_faces})', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.imshow("Face Masks and People Detection" + " -Model: " + model_name, frame)
    key = cv2.waitKey(1) & 0xFF

	#here we add the possibility to press B to end the loop
    if keyboard.is_pressed("b"):
        print("\n<INFO> Ending live video stream.")
        break


cv2.destroyAllWindows()
vs.stop()


