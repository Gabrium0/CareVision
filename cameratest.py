import cv2

def main():
    # Force Camera 1 (your OV2735 UVC device)
    camera_index = 1 
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open Camera 1. Make sure it's plugged in.")
        return

    # Set to its native 2MP resolution for maximum sharpness
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("OV2735 Camera 1 Started Successfully!")
    print("Controls:")
    print("  Press '+' to Zoom In")
    print("  Press '-' to Zoom Out")
    print("  Press 'q' to Quit")

    zoom_factor = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        height, width, _ = frame.shape

        # Digital Zoom Calculations
        if zoom_factor > 1.0:
            new_h, new_w = int(height / zoom_factor), int(width / zoom_factor)
            start_y = (height - new_h) // 2
            start_x = (width - new_w) // 2
            cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
            display_frame = cv2.resize(cropped, (width, height))
        else:
            display_frame = frame

        # UI Overlay
        cv2.putText(display_frame, f"Cam 1 | Zoom: {zoom_factor:.1f}x", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the window (resized slightly just so it fits comfortably on your monitor)
        cv2.imshow('OV2735 Camera Feed', cv2.resize(display_frame, (960, 540)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            zoom_factor = min(zoom_factor + 0.1, 5.0)
        elif key == ord('-'):
            zoom_factor = max(zoom_factor - 0.1, 1.0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()