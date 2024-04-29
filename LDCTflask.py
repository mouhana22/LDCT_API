import threading
from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import Loader
from dicompylercore import dicomparser

app = Flask(__name__)

UPLOAD_FOLDER = './DataInput/'

# Define subdirectories for input and label images
INPUT_FOLDER = 'input'
LABEL_FOLDER = 'label'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OUTPUT_FOLDER = './Output/'

app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def empty_output_folder(folder):
    # Iterate over all files in the OUTPUT_FOLDER and delete them
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def convert_dicom_to_png(dicom_file_path, output_png_path):
    dicom = dicomparser.DicomParser(dicom_file_path)
    png = dicom.GetImage(window=400, level=40, size=None, background=False, frames=0)
    png.save(output_png_path)


@app.route('/denoise', methods=['POST'])
def denoise_image():

    empty_output_folder('./DataInput/input')
    empty_output_folder('./DataInput/label')
    empty_output_folder(OUTPUT_FOLDER)

    # Check if both images are present in the request
    if 'input_image' not in request.files or 'label_image' not in request.files:
        return jsonify({'error': 'Both input and label images are required'})

    input_file = request.files['input_image']
    label_file = request.files['label_image']

    # If any file is not selected, return an error
    if input_file.filename == '' or label_file.filename == '':
        return jsonify({'error': 'Both input and label images must be selected'})

    # Save the files to the UPLOAD_FOLDER
    input_filename = secure_filename(input_file.filename)
    label_filename = secure_filename(label_file.filename)

    input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FOLDER, input_filename)
    label_file_path = os.path.join(app.config['UPLOAD_FOLDER'], LABEL_FOLDER, label_filename)

    input_file.save(input_file_path)
    label_file.save(label_file_path)

    convert_dicom_to_png(input_file_path, './DataInput/input/input_image.png')
    convert_dicom_to_png(label_file_path, './DataInput/label/label_image.png')

    os.remove(input_file_path)
    os.remove(label_file_path)
    # Run your existing code
    Loader.Loader()
    
    # Get the denoised image from the OUTPUT_FOLDER
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f'denoised_image-0.png')

    response = send_file(output_file_path, as_attachment=True)


    def remove_files():
        os.remove('./DataInput/input/input_image.png')
        os.remove('./DataInput/label/label_image.png')
        os.remove(output_file_path)
        

    threading.Timer(1, remove_files).start()

    # Send the denoised image data back to the frontend
    return response



if __name__ == '__main__':
    app.run(debug=True)
