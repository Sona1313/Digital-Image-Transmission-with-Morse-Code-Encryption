import cv2
import numpy as np
import socket
import time
import argparse
import string
import os

def load_sample_image(file_path="/home/sonak/sample.jpeg"):
    """Load a sample image from file instead of capturing"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample image not found at {file_path}")
    
    print(f"Loading sample image from {file_path}...")
    image = cv2.imread(file_path)
    if image is None:
        raise RuntimeError(f"Failed to load image from {file_path}")
    
    print("Sample image loaded successfully!")
    return image

def capture_image():
    """Capture an image from the camera"""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera!")
    
    print("Capturing image...")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Image capture failed!")
    
    print("Image captured successfully!")
    cap.release()
    return frame

def create_histogram(image):
    """Create a histogram of the image"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalize histogram to create a smaller visualization
    hist_height = 50
    hist_width = 256
    hist_img = np.zeros((hist_height, hist_width), dtype=np.uint8)
    
    # Normalize histogram to fit in the image
    cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
    
    # Draw histogram as an image
    for i in range(hist_width):
        cv2.line(hist_img, (i, hist_height), (i, hist_height - int(hist[i])), 255)
    
    print("Histogram created successfully")
    return hist_img

def preprocess_image(image, target_size=(64, 48)):
    """Preprocess the image to reduce size and prepare for encoding"""
    # Resize to target size
    resized = cv2.resize(image, target_size)
    print(f"Image preprocessed to {target_size[0]}x{target_size[1]} dimensions")
    return resized

def encode_pixel_to_morse(pixel):
    """
    Encode an RGB pixel to an alphanumeric character + number that can be represented in Morse code
    - R channel determines letter (A-Z)
    - G channel determines number (0-9)
    - B channel is used as a checksum
    """
    # Map R value (0-255) to letters A-Z (we use modulo to ensure it's in range)
    r_scaled = (pixel[0] % 26)
    letter = string.ascii_uppercase[r_scaled]
    
    # Map G value (0-255) to numbers 0-9
    g_scaled = (pixel[1] % 10)
    number = str(g_scaled)
    
    # Combine letter and number
    alphanumeric = letter + number
    
    # Convert to Morse code
    morse_map = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 
        'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---', 
        '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', 
        '8': '---..', '9': '----.'
    }
    
    # Get Morse code for letter and number
    morse_letter = morse_map[letter]
    morse_number = morse_map[number]
    
    # Combine with a separator
    morse_code = morse_letter + '/' + morse_number
    
    return morse_code

def encode_image_to_morse(image):
    """Encode the entire image to Morse code"""
    height, width = image.shape[:2]
    morse_lines = []
    
    for y in range(height):
        morse_row = []
        for x in range(width):
            if len(image.shape) == 3:  # Color image
                pixel = image[y, x]
            else:  # Grayscale image (histogram)
                pixel = [image[y, x], image[y, x], 0]  # Convert grayscale to pseudo-RGB
            
            morse_pixel = encode_pixel_to_morse(pixel)
            morse_row.append(morse_pixel)
        
        # Join all pixels in a row with a pipe separator
        morse_line = '|'.join(morse_row)
        morse_lines.append(morse_line)
    
    # Join all rows with a custom row separator
    morse_code = '~'.join(morse_lines)
    
    print(f"Image encoded to Morse code: {len(morse_code)} characters")
    return morse_code

def prepare_data_package(image_morse, hist_morse):
    """Combine image and histogram Morse codes into a single package with metadata"""
    # Format: IMAGE_DATA:{width}x{height}:{morse_code}##HIST_DATA:{width}x{height}:{morse_code}
    image_height, image_width = len(image_morse.split('~')), len(image_morse.split('~')[0].split('|'))
    hist_height, hist_width = len(hist_morse.split('~')), len(hist_morse.split('~')[0].split('|'))
    
    package = f"IMAGE_DATA:{image_width}x{image_height}:{image_morse}##HIST_DATA:{hist_width}x{hist_height}:{hist_morse}"
    print(f"Data package prepared: {len(package)} bytes")
    return package

def send_morse_code(morse_code, host='0.0.0.0', port=5000):
    """Send the Morse code to a client"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"Server started on {host}:{port}")
    print("Waiting for client connection...")
    
    client_socket, client_address = server_socket.accept()
    print(f"Connected to client: {client_address}")
    
    # Send the size of the data first
    data_size = len(morse_code)
    data_size_bytes = data_size.to_bytes(8, byteorder='big')
    client_socket.sendall(data_size_bytes)
    
    # Send the Morse code in chunks
    morse_bytes = morse_code.encode('utf-8')
    total_sent = 0
    chunk_size = 1024
    
    while total_sent < data_size:
        chunk = morse_bytes[total_sent:total_sent + chunk_size]
        sent = client_socket.send(chunk)
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        
        total_sent += sent
        print(f"Sent {total_sent}/{data_size} bytes ({total_sent/data_size*100:.2f}%)")
    
    print("Data transmission complete")
    client_socket.close()
    server_socket.close()
    print("Image sender completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Capture and send an image as Morse code")
    parser.add_argument('-p', '--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('-s', '--save', action='store_true', help='Save the original and processed images')
    parser.add_argument('-f', '--file', default="/home/sonak/sample.jpeg", help='Path to sample image file')
    parser.add_argument('-c', '--camera', action='store_true', help='Use camera instead of sample image')
    args = parser.parse_args()
    
    try:
        # Load or capture image
        if args.camera:
            image = capture_image()
        else:
            image = load_sample_image(args.file)
        
        # Save original image if requested
        if args.save:
            cv2.imwrite('original_image.jpg', image)
            print("Saved original image as 'original_image.jpg'")
        
        # Create histogram
        histogram = create_histogram(image)
        
        # Save histogram if requested
        if args.save:
            cv2.imwrite('histogram.jpg', histogram)
            print("Saved histogram as 'histogram.jpg'")
        
        # Preprocess images
        processed_image = preprocess_image(image)
        processed_hist = preprocess_image(histogram, target_size=(256, 50))
        
        # Save processed images if requested
        if args.save:
            cv2.imwrite('processed_image.jpg', processed_image)
            print("Saved processed image as 'processed_image.jpg'")
            cv2.imwrite('processed_histogram.jpg', processed_hist)
            print("Saved processed histogram as 'processed_histogram.jpg'")
        
        # Encode images to Morse code
        image_morse = encode_image_to_morse(processed_image)
        hist_morse = encode_image_to_morse(processed_hist)
        
        # Prepare data package
        data_package = prepare_data_package(image_morse, hist_morse)
        
        # Send Morse code
        send_morse_code(data_package, port=args.port)
        
    except Exception as e:
        print(f"ERROR - {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
