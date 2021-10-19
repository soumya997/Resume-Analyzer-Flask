import re
import os
# import cv2
from flask import Flask, request, redirect, url_for, render_template, send_from_directory

ROOT_DIR = os.getcwd()
FileSaveDir = os.path.join(ROOT_DIR, "uploads")
app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True


@app.route('/form',methods=['GET','POST'])
def form():
    if request.method == 'POST':
        f = request.files['file1']
        FileSavePath = os.path.join(FileSaveDir, f.filename)
        f.save(FileSavePath)

        return render_template('index2.html',message="success")

    return '''
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file1" </input>
            <input type = "submit">
        </form>
    '''

if __name__ == '__main__':
   app.run(debug=True)

