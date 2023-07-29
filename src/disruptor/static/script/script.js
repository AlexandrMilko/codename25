const imageLinks = document.querySelectorAll('.image-picker a'); // getting all links from the page 


// this is function for checking, if one image is alerady picked, and user want to pick another, the effect of picked image from the previous img will disappear
const handleImageClick = (event) => {  
  imageLinks.forEach((link) => link.classList.remove('clicked'));
  event.currentTarget.classList.add('clicked');
}


imageLinks.forEach((link) => {
  link.addEventListener('click', handleImageClick);
});


