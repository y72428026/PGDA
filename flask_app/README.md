# Flask 图像直方图均衡化 API

这是一个基于 Flask 的 Web 应用程序，提供图像直方图均衡化功能。用户可以通过 Web 界面上传图像，系统会自动进行直方图均衡化处理并显示结果。

![Flask App Homepage](https://github.com/user-attachments/assets/de548c68-259b-4bf3-b135-351b6897941c)

## 功能特点

- 🖼️ 支持多种图像格式：PNG, JPG, JPEG, GIF, BMP, TIFF
- 🎨 高质量直方图均衡化处理（LAB色彩空间）
- 🌐 友好的中文Web界面
- 📱 响应式设计，支持移动端
- ⚡ 快速处理和实时预览
- 🔒 安全的文件上传验证

## 环境配置 (Environment Setup)

### 方法一：使用系统包管理器（推荐）

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-flask python3-opencv python3-pil python3-numpy

# CentOS/RHEL
sudo yum install python3-flask python3-opencv python3-pillow python3-numpy

# 或者使用 dnf (较新系统)
sudo dnf install python3-flask python3-opencv python3-pillow python3-numpy
```

### 方法二：使用 Anaconda

```bash
# 创建新的conda环境
conda create -n flask_opencv python=3.8
conda activate flask_opencv

# 安装所需包
conda install flask opencv pillow numpy

# 或者使用conda-forge渠道
conda install -c conda-forge flask opencv pillow numpy
```

### 方法三：使用 pip

```bash
# 创建虚拟环境 (推荐)
python3 -m venv flask_opencv_env
source flask_opencv_env/bin/activate  # Linux/Mac
# 或 flask_opencv_env\Scripts\activate  # Windows

# 安装所需依赖
pip install -r flask_app/requirements.txt
```

## 安装和运行

1. **克隆或下载代码**：
```bash
cd flask_app
```

2. **确保依赖已安装** (见上面的环境配置)

3. **启动Flask服务器**：
```bash
python3 app.py
```

4. **访问Web界面**：
   打开浏览器访问：`http://localhost:5000`

## 使用方法

1. 在Web界面中点击"选择文件"或直接拖拽图片到上传区域
2. 选择要处理的图像文件
3. 点击"处理图像"按钮
4. 等待处理完成，查看原始图像和均衡化后的对比结果

## API 接口

### 端点列表

- **GET /** - 主页面，返回文件上传表单界面
- **POST /equalize** - 图像直方图均衡化处理端点

### API 使用示例

```python
import requests

# 上传并处理图像
with open('your_image.jpg', 'rb') as f:
    files = {'file': ('image.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:5000/equalize', files=files)
    
if response.status_code == 200:
    result = response.json()
    if result['success']:
        # result['image'] 包含处理后的图像(base64编码)
        processed_image = result['image']
        print(result['message'])
    else:
        print('Error:', result['error'])
```

### 响应格式

成功响应：
```json
{
    "success": true,
    "image": "base64_encoded_image_data",
    "message": "图像直方图均衡化处理完成"
}
```

错误响应：
```json
{
    "error": "错误描述信息"
}
```

## 测试

运行测试脚本验证功能：

```bash
# 首先启动服务器
python3 app.py

# 在另一个终端运行测试
python3 test_api.py
```

## 技术细节

- **图像处理算法**：使用OpenCV在LAB色彩空间进行直方图均衡化，保持色彩信息的同时改善亮度分布
- **Web框架**：Flask 3.0.2
- **图像处理**：OpenCV 4.6.0
- **前端技术**：HTML5 + CSS3 + JavaScript (原生)
- **文件限制**：最大16MB，支持常见图像格式

## 文件结构

```
flask_app/
├── app.py              # Flask应用主文件
├── templates/
│   └── index.html     # Web界面模板
├── requirements.txt    # Python依赖列表
├── test_api.py        # API测试脚本
└── README.md          # 说明文档
```

## 故障排除

### 常见问题

1. **依赖包未找到**：
   ```bash
   # 确保所有依赖都已正确安装
   python3 -c "import flask, cv2, PIL, numpy; print('All packages available')"
   ```

2. **端口被占用**：
   ```bash
   # 检查端口使用情况
   netstat -tulpn | grep :5000
   # 或更改app.py中的端口号
   ```

3. **文件上传失败**：
   - 检查文件格式是否支持
   - 确保文件大小不超过16MB
   - 检查服务器日志获取详细错误信息

## 许可证

本项目遵循MIT许可证。