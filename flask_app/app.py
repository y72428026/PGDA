#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API for Image Histogram Equalization
"""

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 允许的图像文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def histogram_equalization(image_path):
    """
    对图像进行直方图均衡化处理
    
    Args:
        image_path (str): 输入图像路径
        
    Returns:
        numpy.ndarray: 处理后的图像数组
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 转换为LAB颜色空间进行直方图均衡化
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道进行直方图均衡化
    cl = cv2.equalizeHist(l)
    
    # 合并通道
    limg = cv2.merge((cl, a, b))
    
    # 转换回BGR颜色空间
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def image_to_base64(image_array):
    """将numpy图像数组转换为base64字符串"""
    # 转换BGR到RGB
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(rgb_image)
    
    # 转换为base64
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/equalize', methods=['POST'])
def equalize_image():
    """
    图像直方图均衡化API端点
    
    Returns:
        JSON响应，包含处理后的图像(base64编码)或错误信息
    """
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式。请上传图像文件。'}), 400
    
    try:
        # 创建临时文件保存上传的图像
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # 执行直方图均衡化
        equalized_image = histogram_equalization(temp_path)
        
        # 转换为base64
        img_base64 = image_to_base64(equalized_image)
        
        # 清理临时文件
        os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'message': '图像直方图均衡化处理完成'
        })
        
    except Exception as e:
        # 清理临时文件
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return jsonify({'error': f'处理图像时出现错误: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件太大，请上传小于16MB的图像文件'}), 413

if __name__ == '__main__':
    # 创建templates目录
    os.makedirs('templates', exist_ok=True)
    
    print("启动Flask服务器...")
    print("访问 http://localhost:5000 来使用图像直方图均衡化功能")
    app.run(host='0.0.0.0', port=5000, debug=True)