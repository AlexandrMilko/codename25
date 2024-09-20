import requests  # Correctly import the requests library
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

from constants import Path

class ImageSegmentor:
    def __init__(self, input_path, output_path, resolution):
        self.resolution = resolution
        self.output_path = output_path
        self.input_path = input_path
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        try:
            server_address = "127.0.0.1:8188"
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        except:
            server_address = 'host.docker.internal:8188'
            req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        try:
            server_address = "127.0.0.1:8188"
            with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
                return response.read()
        except:
            server_address = 'host.docker.internal:8188'
            with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
                return response.read()

    def get_history(self, prompt_id):
        try:
            server_address = "127.0.0.1:8188"
            with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except:
            server_address = 'host.docker.internal:8188'
            with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())

    def get_images(self, ws, prompt):
        print("Queueing prompt...")
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        print(f"Prompt ID: {prompt_id}")

        output_images = {}
        current_node = ""
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    print(f"Received message: {message}")
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['prompt_id'] == prompt_id:
                            print(f"Executing node: {data['node']}")
                            if data['node'] is None:
                                break  # Execution is done
                            else:
                                current_node = data['node']
                else:
                    if current_node == 'save_image_websocket_node':
                        images_output = output_images.get(current_node, [])
                        images_output.append(out[8:])
                        output_images[current_node] = images_output
                        print(f"Image received from node {current_node}")
            except Exception as e:
                print(f"Exception in get_images: {e}")
                break

        return output_images

    def upload_file(self, file, subfolder="", overwrite=False):
        try:
            # Wrap file in form-data so it includes filename
            files = {"image": file}
            data = {}

            if overwrite:
                data["overwrite"] = "true"

            if subfolder:
                data["subfolder"] = subfolder

            try:
                server_address = "127.0.0.1:8188"
                print(f"Uploading file to: http://{server_address}/upload/image")
                resp = requests.post(f"http://{server_address}/upload/image", files=files, data=data)
            except:
                server_address = 'host.docker.internal:8188'
                print(f"Uploading file to: http://{server_address}/upload/image")
                resp = requests.post(f"http://{server_address}/upload/image", files=files, data=data)

            if resp.status_code == 200:
                response_data = resp.json()
                # Add the file to the dropdown list and update the widget value
                path = response_data["name"]
                if "subfolder" in response_data:
                    if response_data["subfolder"] != "":
                        path = response_data["subfolder"] + "/" + path
                print(f"File uploaded successfully: {path}")
                return path  # Ensure path is returned here
            else:
                print(f"Failed to upload file: {resp.status_code} - {resp.reason}")
                return None  # Return None in case of failure
        except Exception as error:
            print(f"Exception during file upload: {error}")
            return None  # Return None in case of exception

    def process_images(self):
        # Load the workflow and set up prompt
        workflow = "preprocessing/segmentor_api.json"
        with open(workflow, "r", encoding="utf-8") as f:
            workflow_data = f.read()
        prompt = json.loads(workflow_data)

        print("Uploading first image...")
        with open(self.input_path, "rb") as f:
            comfyui_path_image = self.upload_file(f, "", True)
            print(f"comfyui_path_image: {comfyui_path_image}")


        # Check if images were uploaded successfully before proceeding
        if not comfyui_path_image:
            print("Image upload failed. Exiting.")
            return

        # Set the image name for our LoadImage node
        print(f"Setting image paths for LoadImage nodes...")
        prompt["24"]["inputs"]["image"] = comfyui_path_image

        prompt["38"]["inputs"]["resolution"] = self.resolution

        prompt["45"]["inputs"]["path"] = self.output_path

        # Connect to the WebSocket server
        ws = websocket.WebSocket()
        try:
            server_address = "127.0.0.1:8188"
            ws.connect(f"ws://{server_address}/ws?clientId={self.client_id}")
            print("Connected to WebSocket server.")
        except:
            server_address = 'host.docker.internal:8188'
            ws.connect(f"ws://{server_address}/ws?clientId={self.client_id}")
            print("Connected to WebSocket server.")

        images = self.get_images(ws, prompt)
        print(f"Received images: {len(images)}")

    def execute(self):
        self.process_images()
