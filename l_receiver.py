#!/usr/bin/env python3
import socket
import time
import argparse
import numpy as np
import cv2
import string
from PIL import Image
import io
import sys
import os
import platform
import subprocess

def ping_host(host):
    """Check if host is reachable via ping"""
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '1', host]
    try:
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return output.returncode == 0
    except Exception:
        return False

def check_port(host, port):
    """Check if a specific port is open on the host"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False
def visualize_morse(received_morse):
    # Convert the received Morse to a visual representation
    visual_map = {'.': '•', '-': '−', ' ': ' '}  # Using proper typographic symbols
    visualized = ''.join(visual_map.get(c, c) for c in received_morse)
    
    # Print with clear formatting
    print("\n=== MORSE CODE RECEIVED ===")
    print(visualized)
    print("=========================\n")
    
    # If you also want to show the translation
    translated = morse_to_text(received_morse)
    print(f"Translated text: {translated}")

def decode_morse_to_alphanumeric(morse_code):
    """Decode Morse code to alphanumeric characters"""
    # Morse code to alphanumeric mapping
    morse_map = {
        '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
        '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
        '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
        '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
        '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
        '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
        '---..': '8', '----.': '9'
    }
    
    # Split into letter and number parts
    parts = morse_code.split('/')
    if len(parts) != 2:
        return None  # Invalid format
    
    letter_morse, number_morse = parts
    
    # Decode letter and number
    if letter_morse in morse_map and number_morse in morse_map:
        letter = morse_map[letter_morse]
        number = morse_map[number_morse]
        return letter + number
    
    return None  # Invalid Morse code

def decode_alphanumeric_to_pixel(alphanumeric):
    """Convert alphanumeric code back to an RGB pixel"""
    if len(alphanumeric) != 2:
        return (0, 0, 0)  # Default black pixel for invalid codes
    
    letter = alphanumeric[0]
    number = alphanumeric[1]
    
    # Convert letter (A-Z) to R value (0-255 range)
    r_value = (ord(letter) - ord('A')) * 10  # Scale for better visibility
    
    # Convert number (0-9) to G value (0-255 range)
    g_value = int(number) * 28  # Scale for better visibility
    
    # Generate B value as a function of R and G
    b_value = (r_value + g_value) % 256
    
    return (r_value, g_value, b_value)

def decode_morse_to_image(morse_text, image_size=(64, 48)):
    """Decode Morse code text back to an image"""
    width, height = image_size
    
    # Split the whole Morse code into rows
    morse_rows = morse_text.split('~')
    
    # Check if we have the expected number of rows
    if len(morse_rows) != height:
        print(f"Warning: Expected {height} rows, but got {len(morse_rows)}")
    
    # Create an empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process each row
    for y, morse_row in enumerate(morse_rows[:height]):
        # Split the row into pixels
        morse_pixels = morse_row.split('|')
        
        # Check if we have the expected number of pixels in this row
        if len(morse_pixels) != width:
            print(f"Warning: Row {y}: Expected {width} pixels, but got {len(morse_pixels)}")
        
        # Process each pixel
        for x, morse_pixel in enumerate(morse_pixels[:width]):
            try:
                # Decode Morse code to alphanumeric
                alphanumeric = decode_morse_to_alphanumeric(morse_pixel)
                
                if alphanumeric:
                    # Convert alphanumeric to RGB
                    pixel = decode_alphanumeric_to_pixel(alphanumeric)
                    image[y, x] = pixel
            except Exception as e:
                print(f"Error decoding pixel at ({x},{y}): {e}")
    
    # Resize for better viewing (optional)
    display_image = cv2.resize(image, (width*4, height*4), interpolation=cv2.INTER_NEAREST)
    
    # Convert to PIL Image for compatibility
    pil_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    
    print("Image reconstruction successful!")
    return pil_image

def parse_data_package(data_package):
    """Parse the data package containing both image and histogram data"""
    try:
        # Split package into image and histogram parts
        parts = data_package.split('##')
        
        # Parse image data
        image_part = parts[0].strip()
        if not image_part.startswith('IMAGE_DATA:'):
            raise ValueError("Image data section not found in the data package")
        
        image_content = image_part[len('IMAGE_DATA:'):]
        image_meta, image_morse = image_content.split(':', 1)
        image_width, image_height = map(int, image_meta.split('x'))
        
        # Parse histogram data
        if len(parts) < 2:
            print("Warning: Histogram data not found in the data package")
            hist_width, hist_height, hist_morse = 0, 0, ""
        else:
            hist_part = parts[1].strip()
            if not hist_part.startswith('HIST_DATA:'):
                raise ValueError("Histogram data section has invalid format")
            
            hist_content = hist_part[len('HIST_DATA:'):]
            hist_meta, hist_morse = hist_content.split(':', 1)
            hist_width, hist_height = map(int, hist_meta.split('x'))
        
        return (image_width, image_height, image_morse), (hist_width, hist_height, hist_morse)
    
    except Exception as e:
        print(f"Error parsing data package: {e}")
        # Fallback to treating entire package as image data
        return (64, 48, data_package), (0, 0, "")

def save_morse_to_file(morse_text, filename):
    """Save the Morse code text to a file"""
    try:
        with open(filename, 'w') as f:
            f.write(morse_text)
        print(f"Morse code saved to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save Morse code: {e}")
        return False

def receive_morse_image(host='192.168.37.76', port=5000, debug=False):
    """Receive an image transmitted as Morse code from the specified server"""
    # Troubleshooting checks
    if debug:
        print(f"Debug: Checking if host {host} is reachable...")
        if ping_host(host):
            print(f"Debug: Host {host} is reachable via ping")
        else:
            print(f"Debug: WARNING - Host {host} is NOT reachable via ping")
        
        print(f"Debug: Checking if port {port} is open on {host}...")
        if check_port(host, port):
            print(f"Debug: Port {port} is open on {host}")
        else:
            print(f"Debug: WARNING - Port {port} is NOT open on {host}")
    
    # Create a socket and connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(30)  # Set timeout to 30 seconds
    print(f"Connecting to {host}:{port}...")
    
    try:
        client_socket.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        
        # Receive the data size first
        data_size_bytes = client_socket.recv(8)
        if not data_size_bytes:
            print("Error: No data received from server")
            return
            
        data_size = int.from_bytes(data_size_bytes, byteorder='big')
        print(f"Expecting {data_size} bytes of Morse code data")
        
        # Receive the Morse code data
        received_data = b''
        bytes_received = 0
        last_progress = -1
        
        while bytes_received < data_size:
            try:
                chunk = client_socket.recv(4096)
                if not chunk:
                    print("Connection closed prematurely by the server")
                    break
                
                received_data += chunk
                bytes_received = len(received_data)
                progress = int(bytes_received/data_size*100)
                
                # Only print progress updates when the percentage changes
                if progress != last_progress:
                    print(f"Received {bytes_received}/{data_size} bytes ({progress}%)")
                    last_progress = progress
            except socket.timeout:
                print("Timeout while receiving data. Continuing...")
                continue
        
        if bytes_received >= data_size:
            print("Data reception complete")
            
            # Convert Morse code back to image
            morse_text = received_data.decode('utf-8')
            print(f"Received {len(morse_text)} Morse code characters")
            
            # Save the raw morse code to a file
            timestamp = int(time.time())
            morse_filename = f"received_morse_{timestamp}.txt"
            save_morse_to_file(morse_text[:1000] + "... (truncated)", morse_filename)
            
            # Parse the data package to extract image and histogram
            (image_width, image_height, image_morse), (hist_width, hist_height, hist_morse) = parse_data_package(morse_text)
            
            # Process the image Morse code
            image_size = (image_width, image_height)
            decoded_image = decode_morse_to_image(image_morse, image_size)
            
            # Save the received image
            image_filename = f"received_image_{timestamp}.jpg"
            decoded_image.save(image_filename)
            print(f"Image saved as '{image_filename}'")
            
            # Process the histogram Morse code if available
            if hist_width > 0 and hist_height > 0 and hist_morse:
                hist_size = (hist_width, hist_height)
                decoded_hist = decode_morse_to_image(hist_morse, hist_size)
                
                # Save the received histogram
                hist_filename = f"received_histogram_{timestamp}.jpg"
                decoded_hist.save(hist_filename)
                print(f"Histogram saved as '{hist_filename}'")
                
                # Create a side-by-side display of image and histogram
                combined_width = decoded_image.width + decoded_hist.width
                combined_height = max(decoded_image.height, decoded_hist.height)
                combined_image = Image.new('RGB', (combined_width, combined_height))
                combined_image.paste(decoded_image, (0, 0))
                combined_image.paste(decoded_hist, (decoded_image.width, 0))
                
                # Save the combined image
                combined_filename = f"combined_image_{timestamp}.jpg"
                combined_image.save(combined_filename)
                print(f"Combined image and histogram saved as '{combined_filename}'")
                
                # Show the combined image
                if platform.system() == 'Windows':
                    try:
                        combined_image.show()
                    except Exception as e:
                        print(f"Warning: Could not display image: {e}")
                        print("Images were saved and can be viewed manually.")
            else:
                # Just show the image if histogram is not available
                if platform.system() == 'Windows':
                    try:
                        decoded_image.show()
                    except Exception as e:
                        print(f"Warning: Could not display image: {e}")
                        print("Image was saved and can be viewed manually.")
            
            # Show the full paths to the saved files
            print("\nSaved files:")
            for filename in [image_filename, hist_filename if 'hist_filename' in locals() else None, 
                            combined_filename if 'combined_filename' in locals() else None,
                            morse_filename]:
                if filename:
                    abs_path = os.path.abspath(filename)
                    print(f"- {os.path.basename(filename)}: {abs_path}")
            
            print("\nImage receiver completed successfully")
            
        else:
            print(f"Warning: Incomplete data received. Got {bytes_received} of {data_size} bytes")
            
            # Try to reconstruct partial image if enough data was received
            if bytes_received > data_size * 0.5:  # At least 50% received
                print("Attempting to reconstruct partial image from incomplete data...")
                try:
                    morse_text = received_data.decode('utf-8')
                    
                    # Parse partial data
                    try:
                        (image_width, image_height, image_morse), _ = parse_data_package(morse_text)
                        partial_image = decode_morse_to_image(image_morse, (image_width, image_height))
                    except:
                        # Fallback to default size
                        partial_image = decode_morse_to_image(morse_text)
                    
                    partial_filename = f"partial_image_{int(time.time())}.jpg"
                    partial_image.save(partial_filename)
                    print(f"Partial image saved as '{partial_filename}'")
                except Exception as e:
                    print(f"Could not reconstruct partial image: {e}")
        
    except socket.gaierror:
        print(f"Error: Cannot resolve hostname '{host}'. Check your network connection or try using an IP address.")
    except ConnectionRefusedError:
        print(f"Error: Connection refused. Make sure the server is running at {host}:{port}")
        if debug:
            print("Debug: This typically means nothing is listening on the specified port.")
            print("Debug: Check if the server script is running on the Raspberry Pi.")
    except socket.timeout:
        print(f"Error: Connection timed out. Server at {host}:{port} might be unreachable.")
    except Exception as e:
        print(f"Error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    finally:
        client_socket.close()
        print("Socket connection closed")

def main():
    parser = argparse.ArgumentParser(description='Receive an image transmitted as Morse code')
    parser.add_argument('-s', '--host', dest='host', default='192.168.37.76', 
                        help='Server hostname or IP address')
    parser.add_argument('-p', '--port', dest='port', type=int, default=5000, 
                        help='Server port')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--save-morse', dest='save_morse', action='store_true',
                        help='Save received Morse code to a file')
    args = parser.parse_args()
    
    # Start the receiver
    receive_morse_image(args.host, args.port, args.debug)

if __name__ == "__main__":
    main()