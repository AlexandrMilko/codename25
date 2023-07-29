const fileInput = document.getElementById('fileInput');
    const uploadedImage = document.getElementById('uploadedImage');

    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.addEventListener('load', function () {
            const formData = new FormData();
            formData.append('image', file);

            fetch('/save_image', {
                method: 'POST',
                body: formData,
            })
                .then(response => response.json()) // Parse the JSON response
                .then(data => {
                    const url = data.url; // Extract the "url" key from the response
                    uploadedImage.src = url;
                    uploadedImage.style.display = 'block';
                    uploadedImage.style.opacity = 1;
                    console.log("URL of the saved image:", url);
                })
                .catch(error => console.error('Error:', error));
        });

        if (file) {
            reader.readAsDataURL(file);
        }
    });