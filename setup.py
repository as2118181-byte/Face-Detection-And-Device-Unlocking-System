from setuptools import setup, find_packages

setup(
    name="ai-secure-face-authentication",
    version="1.0.0",
    author="Arun Sharma",
    author_email="as2118181@gmail.com",
    description="AI-powered Face Recognition and Liveness Detection System for Secure Authentication",
    long_description="A real-time face authentication system using deep learning, ONNX models, OpenCV, and anti-spoofing techniques.",
    url="",
    license="MIT",
    keywords=[
        "face recognition",
        "liveness detection",
        "anti spoofing",
        "computer vision",
        "opencv",
        "deep learning",
        "onnx",
        "authentication",
        "artificial intelligence"
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)