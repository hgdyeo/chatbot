
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


s_img = cv2.imread("images/hat.jpg", -1)
v_img = cv2.imread("images/beard.jpg", -1)

class Camera():
    def __init__(self):
        self.video  = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        success, frame = self.video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            crop_img = frame[x:x + w, y:y + w]

            im = image_resize(s_img, width=w)

            y_adjust = -30

            y1, y2 = max(y - im.shape[0] - y_adjust, 0), y - y_adjust
            x1, x2 = x, x + im.shape[1]

            alpha_s = (im[:, :, :] / 255).astype(int)
            alpha_l = 1.0 - alpha_s.astype(int)

            for c in range(3):
                frame[y1:y2, x1:x2, c] = alpha_l[-(y2 - y1):, :, c] * im[-(y2 - y1):, :, c] + alpha_s[-(y2 - y1):, :,
                                                                                              c] * frame[y1:y2, x1:x2,
                                                                                                   c]

            im2 = image_resize(v_img, width=w)

            y_adjust = 150

            y1, y2 = y + w - im2.shape[0] + y_adjust, min(y + w + y_adjust, 480)
            x1, x2 = x, x + im2.shape[1]

            alpha_s = (im2[:, :, :] / 255).astype(int)
            alpha_l = 1.0 - alpha_s.astype(int)

            im_y2 = min(640, y2 + im2.shape[1]) - y2

            for c in range(3):
                frame[y1:y2, x1:x2, c] = alpha_l[:y2 - y1, :, c] * im2[:y2 - y1, :, c] + alpha_s[:y2 - y1, :,
                                                                                         c] * frame[
                                                                                              y1:y2,
                                                                                              x1:x2, c]

        ret, jpeg = cv2.imencode('.jpg', frame)

        frame = jpeg.tobytes()

        return frame


def gen():

    camera = Camera()

    while True:
        frame = camera.get_frame()

        yield ( b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
