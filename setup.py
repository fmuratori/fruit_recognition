from setuptools import setup, find_packages

setup(
    name = "fruit-recognition",
    version = "1.0.0",
    packages = find_packages("src"),
    package_dir = {"": "src"},
    include_package_data = True,
    install_requires = [
        "numpy==1.18.4",
        "opencv-python==4.1.2.30",
        "pandas==1.0.3",
        "scikit-learn==0.22.2.post1",
        "tensorflow==2.2.0",
        "Keras==2.3.1",
        "click==7.1.2"
    ],
    entry_points = {
        "console_scripts": [
            "fruit-recognition = fruit_recognition.main:main"
        ]
    }
)
