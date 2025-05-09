# PPE Compliance Monitoring Toolkit

This repository contains a toolkit for monitoring Personal Protective Equipment (PPE) compliance using computer vision and deep learning. It is designed to assist safety officers and industry supervisors in automatically identifying whether individuals are wearing mandatory PPE—such as masks, gloves, hairnet, goggles and full_body suit—within a given video, image or a live stream.

The system leverages a custom-trained YOLO (You Only Look Once) model for real-time object detection and is deployed using a Streamlit-based web interface. Users can upload video footage or image or use live stream from industrial or food production environments, and the toolkit will analyze it frame-by-frame, highlight PPE violations, and generate a summary report in PDF format for documentation and compliance auditing.


![Dashboard View 1](https://github.com/M-Muddassir-Saleem/PPE-Compliance-Monitoring/blob/ea951cf2e21871827c1e63081c69b1a251132ed6/dashboard.png)
<p align="center"><em>(Main dashboard view showing compliance overview)</em></p>



![Dashboard View 1](https://github.com/M-Muddassir-Saleem/PPE-Compliance-Monitoring/blob/70729f1f52c23f815dc7519fadd62cec6329d36e/dashboard%20(1).png)
<p align="center"><em>(Main dashboard view showing safety observation card)</em></p>      



![Dashboard View 1](https://github.com/M-Muddassir-Saleem/PPE-Compliance-Monitoring/blob/f6ca62bc93b98a2e33ff8f192067e25b13c13116/dashboard%20(2).png)
<p align="center"><em>(Observation description)</em></p> 


## Project Overview

The toolkit leverages computer vision and deep learning to ensure that workers in the food industry are wearing the necessary PPE. The model is designed to detect whether the required PPE is being worn in various work environments, helping to enhance safety and regulatory compliance.

## Files Included

- **app.py**: The main application file that runs the PPE compliance monitoring system.
- **requirements.txt**: This contains necessary dependencies.
- **my_model.pt**: A trained PyTorch model for PPE detection.
- **logo.png**: Logo image for the toolkit.

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/M-Muddassir-Saleem/PPE-Compliance-Monitoring.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, feel free to reach out at [muddassirmddassir@gmail.com](mailto:muddassirmddassir@gmail.com).
