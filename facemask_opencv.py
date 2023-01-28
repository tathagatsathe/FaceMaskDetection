import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('./facemask_keras_model.h5')

vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', frame)
    frame = cv2.resize(frame, (128, 128))
    frame = img_to_array(frame)*.1/255

    predictions = model.predict(frame.reshape((1, 128, 128, 1)))
    predictions = (predictions>0.5).astype('int32')
    print(predictions)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()
    