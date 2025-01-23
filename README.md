# Team
## IT guys:
- **[Mykhailo Lapko](https://www.linkedin.com/in/mykhailo-lapko-6928a726b/)**: Motivated DevOps, developed a boatload of algorithms for the app. Pure fun and pleasure to work with.
- **[Illia Martynov](https://www.linkedin.com/in/illia-martynov-335800283/)**: Pentester with passion(and Frontender in past), always knows how to support and speedrunned the frontend tasks.
- **[Arsen Rudnytskyi](https://www.linkedin.com/in/arsen-rudnytskyi/)**: Machine Learning guy thanks to whom you can read our code. Excellent at refactoring and organizing information.
- **[Sviatoslav Pylyp](https://www.linkedin.com/in/sviatoslav-pylyp-3a60b8261/)**: Greatest Backender I've worked with. Need a backend, devops solution? Describe a task in couple of words, he will deliver perfectly. Did it for the startup using clean and maintainable APIs in Java, microservice infrastructure, and hybrid infrastructure leveraging AWS.
- **[Volodymyr Lapkin](https://www.linkedin.com/in/volodymyr-lapkin-5188a92b0/)**: Algorithm engineer. Learns how to work in new environment quickly. He will change the world, give him time.
- **[Kyrylo Turchyn](https://www.linkedin.com/in/kyrylo-turchyn-aa198a248/)**: Machine Learning engineer. Worked outsourced, so you dont see his commits. Helped a lot though with 3D estimation. He is driven to learn AI to train ginormous revolutionary models. Just give him oportunity.
- **[Oleksandr Milko](https://www.linkedin.com/in/oleksandrmilko/)**: Guess we'll find out some day. Cheers!
## Business guys:
- **[Dawid Tambor](https://www.linkedin.com/in/dawid-tambor/)**: The man thanks to whom I started doing this project. He's been a good mentor for me.
- **[Agnieszka Solarz](https://www.linkedin.com/in/agnieszka-solarz/)**: Managed all the work with Designing the app and more. Always brought good energy to our meetings even on the hardest days.
- **[Marcin Mika](https://www.linkedin.com/in/marcin-mika-69656a140/)**: Helped significantly with accounting projections and bureaucracy. Realist, makes conclusions based on facts and statistics.
</br>

# Documentation

An application for virtual staging of rooms(empty room photos). 
- **75% accuracy** from the last test(on simple bedroom layouts)! :sunglasses:
- Takes only **2 minutes**(per picture) to stage a room.
- Uses **real 3D models**. Not an ordinary GenAI. </br>
</br>
We made the app use real-world 3D furniture models because it is the right step toward what was desired by many interior designers.
</br></br>

This application consists of several components that work together to create a staged room image based on an empty room input and style preferences. Below is an overview of the key parts and installation steps.

Interesting links:

0) [link to Miro Management Board(since Summer 2024)](https://miro.com/app/board/uXjVKGo-O6M=/?share_link_id=615622177572)
1) [link to Miro flowchart](https://miro.com/app/board/uXjVLV9t9UI=/?share_link_id=722076167204)
2) [Last tests from November - December of 2024 for Bedrooms](https://drive.google.com/drive/folders/1h40hiGPe5YR-0qQE-AkL4Wa5tM2Z_DqJ?usp=drive_link)
3) [Some of the good results we got with postprocessing](https://drive.google.com/drive/folders/1RD0QD8b955mVSouCyZdUKc254pT331rT?usp=drive_link)
4) links to Backend DockerHub and Github(Made by Sviatoslav): [auth-service](https://hub.docker.com/r/1aughingbird543/auth-service), [email-service](https://hub.docker.com/r/1aughingbird543/email-service), [api-gateway](https://hub.docker.com/r/1aughingbird543/api-gateway), [ai-queue](https://hub.docker.com/r/1aughingbird543/ai-queue), [Backend Github](https://github.com/Codename25)
5) links to Frontend(Made by Illia): [illiamartynov/startup](https://github.com/illiamartynov/startup)
## Application Components

1. **codename25 Repo**  
   The core of the application, serving as the API for image creation.  
   **Input:**
   
   Base64 encoded empty room image, style and budget string in the format:  
   `"{Style}, {Budget}"`  
   Room type string in the format:  
   `"{bedroom/living room/kitchen}"`
   
   **Output:** Base64 encoded staged room image.
   
2. **Blender 3D furniture models**

3. **Stable Diffusion**  
   Handles preprocessing (segmentation) and postprocessing (image enhancement). You can use either **WebUI** or **ComfyUI** for this task.  
   - **WebUI**: Superior for postprocessing. Arguably easier to install.
   - **ComfyUI**: Manages GPU memory more efficiently but is less effective at postprocessing.

4. **Conda Environment**  
   Contains all the necessary packages for the application to function correctly.

# Installing each component
### Conda Environment and CUDA

1. Install **CUDA**: Follow the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
2. Install **Conda**: Download and install Miniconda from [here](https://docs.anaconda.com/miniconda/).
3. Create a new Conda environment: `conda create -n app python=3.11.9 anaconda`
4. Install PyTorch for AI(Dont forget to activate conda env with `conda activate app` beforehand): `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
5. `pip install -r requirements.txt`

### Blender models
Unfortunately, we cannot provide pre-existing 3D models due to licensing restrictions. Follow the steps below.

1. Find or create your own 3d models in USDC format.
2. Test models in Blender to confirm they work as intended, including textures, scaling, and positioning.
3. Prepare the Folder Structure <br />
Navigate to the `Projects/codename25/visuals/3Ds/` directory.
Identify the appropriate category folder for your model (e.g., `living_room` for living room assets).
If the category does not exist, create a new folder with a relevant name (e.g., `bedroom`, `living_room`).
4. Place the .usdc file in the appropriate category folder. <br />
   ![image](https://github.com/user-attachments/assets/854c5add-77bc-424f-b0e4-24df246d121c)
   ![image](https://github.com/user-attachments/assets/53e2ddaf-dfd8-45c3-9d1c-5bf40e6b8703)



### APP itself (codename25)
1. `git clone https://github.com/AlexandrMilko/codename25.git`
2. Put [depth_pro.pt](https://drive.google.com/drive/u/0/folders/1Kg9j__fVpCMmvZ4Bt6jCDhKo3KH98ZW3) model into the folder `codename25/ml_depth_pro/src/depth_pro/cli/checkpoints/`
3. Create folder `codename25\ml_depth_pro\output`
4. If you have macbook: `pip install git+https://github.com/rwightman/pytorch-image-models.git`

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

## Test
To test the API, activate the conda environment and run the test script:
```bash
conda activate app
python test_api.py  # (WARNING: Put the right image path in test_api.py)
```


## How to work with Docker on the Server
1. Create your container
```bash
sudo docker run -it --name vistagerVova --network host --gpus all oleksandrmilko/vistager:demo1309
```
2. (Just in case):
```bash
sudo docker restart vistagerVova
```
3. [Activate github token](https://stackoverflow.com/questions/18935539/authenticate-with-github-using-a-token ) for the git in the container if it is expired
4. Enter a session in container
```bash
sudo docker exec -it vistagerVova bash
```
5. Copy files(and images for example) from docker:
```bash
sudo docker cp <docker_container>:<path_in_docker> <path_in_host_system>
```
## Special thanks
- [ml_depth_pro](https://github.com/apple/ml-depth-pro): for depth estimation to calculate 3D structure of rooms
- [segment_anything](https://github.com/facebookresearch/segment-anything): for masking neighbouring rooms
