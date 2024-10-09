# Documentation

This application consists of several components that work together to create a staged room image based on an empty room input and style preferences. Below is an overview of the key parts and installation steps.
[link to flowchart](https://miro.com/app/board/uXjVLV9t9UI=/)

## Application Components

1. **codename25 Repo**  
   The core of the application, serving as the API for image creation.  
   **Input:**
   
   Base64 encoded empty room image, style and budget string in the format:  
   `"{Style}, {Budget}"`  
   Room type string in the format:  
   `"{bedroom/living room/kitchen}"`
   
   **Output:** Base64 encoded staged room image.
   
3. **Blender Server**  
   This server renders the furniture and 3D space for codename25.

4. **Stable Diffusion**  
   Handles preprocessing (segmentation) and postprocessing (image enhancement). You can use either **WebUI** or **ComfyUI** for this task.  
   - **WebUI**: Superior for postprocessing. Arguably easier to install.
   - **ComfyUI**: Manages GPU memory more efficiently but is less effective at postprocessing.

5. **Conda Environment**  
   Contains all the necessary packages for the application to function correctly.

# Installing each component
### Conda Environment and CUDA

1. Install **CUDA**: Follow the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
2. Install **Conda**: Download and install Miniconda from [here](https://docs.anaconda.com/miniconda/).
3. Create a new Conda environment: `conda create -n app python=3.11.9 anaconda`
4. Install PyTorch for AI(Dont forget to activate conda env with `conda activate app` beforehand): `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
5. `pip install -r requirements.txt`

### Blender

1. Install [Blender 4.2](https://www.blender.org/)
2. Clone the Blender server: `git clone https://github.com/AlexandrMilko/codename25other.git`
3. Download [3D models](https://drive.google.com/drive/folders/1Bq_OSmUj9t5iwL2zB5yfb1n_nRBtqSVf?usp=sharing) and put them instead of the folder 3Ds
4. Add blender to PATH environment variable(`C:\Program Files\Blender Foundation\Blender 4.2` is the default path for Windows)

### APP itself (codename25)
1. `git clone https://github.com/AlexandrMilko/codename25.git`
2. Create folder `codename25\images\preprocessed`
3. Put [Depth Anything v2](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true) model into the folder `codename25\DepthAnythingV2\metric_depth\checkpoints`
4. Create folder `codename25\DepthAnythingV2\output`

### Stable Diffusion
1. Install [WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Install Controlnet through extensions
![controlnet](https://github.com/user-attachments/assets/c4a426b2-7f0d-4079-b00e-f755b3004e99)
3. Download and put the [controlnet models](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main)(control_v11f1p_sd15_depth.pth, control_v11p_sd15_seg.pth) in the directory: `extensions/sd-webui-controlnet/models`
4. Download and put [stable diffusion model](https://civitai.com/models/4201/realistic-vision-v60-b1) in `models/Stable-diffusion`
5. Replace webui.py with the [webui.py from codename25other](https://github.com/AlexandrMilko/codename25other/blob/master/webui.py)

# Running the App

## Web UI (First run the WEBUI! The app will check connection to it on the start)
**IMPORTANT:** change the `ui` parameter in [constants.py](https://github.com/AlexandrMilko/codename25/blob/main/constants.py) to 'webui'

Run the following command:
```bash
webui.bat --nowebui --server-name=0.0.0.0
```
Or if you run UNIX:
```bash
./webui.sh --nowebui --server-name=0.0.0.0
```
## App
Next, activate the conda environment and run the application:
```bash
conda activate app
cd codename25
python run.py
```

## Blender
Activate the conda environment and run the server:
```bash
conda activate app
python server.py
```

## Test
To test the API, activate the conda environment and run the test script:
```bash
conda activate app
python test_api.py  # (WARNING: Put the right image path in test_api.py)
```

