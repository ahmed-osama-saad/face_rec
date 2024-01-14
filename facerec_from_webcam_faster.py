import face_recognition
import cv2
import numpy as np

from datetime import datetime, timedelta
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = 'https://yxgxrywwaawceiomwsvp.supabase.co'
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl4Z3hyeXd3YWF3Y2Vpb213c3ZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTc5OTc3NTEsImV4cCI6MjAxMzU3Mzc1MX0.uUhu_pheoHoTQHF1-HhygUcshkDCzp0IMqNwbvUn24I"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# A dictionary to hold the last recognized timestamp for each employee
last_recognized = {}

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
# video_capture = cv2.VideoCapture(0)


stream_url = "rtsp://admin:pass%402468@192.168.1.201:554/Streaming/Channels/501"
video_capture = cv2.VideoCapture(stream_url)
# i want to load all images from the folder faces and get the names of the people from the file names


# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

oss_image = face_recognition.load_image_file("oss.jpg")
oss_face_encoding = face_recognition.face_encodings(oss_image)[0]

zeez_image = face_recognition.load_image_file("zeez.jpg")
zeez_face_encoding = face_recognition.face_encodings(zeez_image)[0]

radwa_image = face_recognition.load_image_file("radwa.jpg")
radwa_face_encoding = face_recognition.face_encodings(radwa_image)[0]

walid_image = face_recognition.load_image_file("walid.jpg")
walid_face_encoding = face_recognition.face_encodings(walid_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    oss_face_encoding,
    zeez_face_encoding,
    radwa_face_encoding,
    walid_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Oss",
    "Zeez",
    "Radwa",
    "Walid"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            if name != "Unknown":
                print(name)
                now = datetime.now()

                # Check the duration since the last recognition
                last_time = last_recognized.get(name)
                if not last_time or (now - last_time) > timedelta(minutes=30):  # Adjust the duration as needed
                    # Log this event to Supabase
                    data = supabase.table('employee_logs').insert(
                        {
                            "employee_name": name,
                        }
                    ).execute()

                    assert len(data.data) > 0
                    last_recognized[name] = now

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
