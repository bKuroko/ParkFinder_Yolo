import cv2
import json


# =========================================
# SETTINGS
# =========================================

VIDEO_PATH = "videos/parking.mp4"

FRAME_TIME_MS = 0
# 0 = first frame
# 5000 = 5 seconds
# 10000 = 10 seconds

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# =========================================
# STORAGE
# =========================================

points = []
parking_spaces = []


# =========================================
# MOUSE CALLBACK
# =========================================

def mouse_callback(event, x, y, flags, param):

    global points
    global parking_spaces

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"Point selected: ({x}, {y})")

        # Every 4 clicks = one parking space
        if len(points) == 4:

            parking_spaces.append(
                points.copy()
            )

            print(
                f"Parking space A{len(parking_spaces)} created."
            )

            points = []


# =========================================
# OPEN VIDEO
# =========================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        f"ERROR: Could not open {VIDEO_PATH}"
    )

    exit()


# =========================================
# SELECT FRAME FROM VIDEO
# =========================================

cap.set(
    cv2.CAP_PROP_POS_MSEC,
    FRAME_TIME_MS
)

success, frame = cap.read()

cap.release()


if not success or frame is None:

    print(
        "ERROR: Could not read frame from parking video."
    )

    exit()


# =========================================
# RESIZE FRAME
# =========================================

frame = cv2.resize(
    frame,
    (
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )
)


print(
    f"Reference frame size: "
    f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}"
)


# =========================================
# CREATE WINDOW
# =========================================

window_name = "Parking Space Selector"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# =========================================
# MAIN LOOP
# =========================================

while True:

    display = frame.copy()


    # =====================================
    # DRAW SAVED PARKING SPACES
    # =====================================

    for index, space in enumerate(
        parking_spaces
    ):

        # Draw polygon lines
        for i in range(4):

            start = space[i]

            end = space[
                (i + 1) % 4
            ]

            cv2.line(
                display,
                start,
                end,
                (0, 255, 0),
                2
            )


        # =================================
        # PARKING SPACE LABEL
        # =================================

        x, y = space[0]

        cv2.putText(
            display,
            f"A{index + 1}",
            (
                x,
                max(25, y - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # =====================================
    # DRAW CURRENT UNSAVED POINTS
    # =====================================

    for point in points:

        cv2.circle(
            display,
            point,
            6,
            (0, 0, 255),
            -1
        )


    # =====================================
    # INSTRUCTIONS
    # =====================================

    cv2.rectangle(
        display,
        (10, 10),
        (650, 55),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        "Click 4 corners | S=Save | R=Reset | Q=Quit",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =====================================
    # SHOW WINDOW
    # =====================================

    cv2.imshow(
        window_name,
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # =====================================
    # SAVE
    # =====================================

    if key == ord("s"):

        if len(parking_spaces) == 0:

            print(
                "WARNING: No parking spaces selected."
            )

            continue


        data = {}

        for index, space in enumerate(
            parking_spaces
        ):

            data[
                f"A{index + 1}"
            ] = space


        with open(
            "data/parking_spaces.json",
            "w"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )


        print(
            "Saved to data/parking_spaces.json"
        )

        print(
            f"Total parking spaces: "
            f"{len(parking_spaces)}"
        )


    # =====================================
    # RESET
    # =====================================

    elif key == ord("r"):

        points = []
        parking_spaces = []

        print(
            "All parking spaces cleared."
        )


    # =====================================
    # QUIT
    # =====================================

    elif key == ord("q"):

        print(
            "Closing parking selector."
        )

        break


# =========================================
# CLEANUP
# =========================================

cv2.destroyAllWindows()