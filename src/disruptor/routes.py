from flask import redirect, render_template, url_for, flash, request, jsonify
from disruptor.forms import RegistrationForm, LoginForm
from disruptor import app, db
from flask_bcrypt import generate_password_hash, check_password_hash
from disruptor.models import User, load_user
from flask_login import (
    login_required,
    login_user,
    logout_user,
)
from disruptor.google_auth import *
from disruptor.sdquery import text_query, image_query, set_deliberate
import base64
import json
import uuid
from sqlalchemy.exc import IntegrityError


@app.route("/")
@app.route("/home")
def home():
    return render_template('homepage.html', title="Home")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        password_hash = generate_password_hash(form.password.data).decode("utf-8")
        uuid_ = str(uuid.uuid4())
        if not load_user(uuid_): # We generate the uuid again if such a user exists with the same id. Just in case. Highly-unlikely
            user = User(
                id=uuid_,
                username=form.username.data,
                email=form.email.data,
                password=password_hash
                        )
        else:
            uuid_ = str(uuid.uuid4())
            user = User(
                id=uuid_,
                username=form.username.data,
                email=form.email.data,
                password=password_hash
            )
        db.session.add(user)
        db.session.commit()
        flash(f"Account created for {form.username.data}!", 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title="Register", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        try:
            if user and check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                flash(f"Successfully logged into {user.username}", 'success')
                next_page = request.args.get('next')
                return redirect(url_for(next_page)) if next_page else redirect(url_for('home'))
            else:
                flash(f"Wrong email or password", 'danger')
        except TypeError:
            flash(f"Such an account has already been created via Google!", 'danger')
            return redirect(url_for("login"))
    return render_template('login.html', title="Login", form=form)

@app.route("/login-google")
def login_google():
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Use library to construct the request for Google login and provide
    # scopes that let you retrieve user's profile from Google
    # I want to remove /login-google from the link, thus I remove last element after splitting it with /
    redirect_uri = "/".join(request.base_url.split("/")[:-1]) + url_for("callback") # http://localhost:5000/login-google/callback -> http://localhost:5000/callback
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route("/callback")
def callback():
    # Get authorization code Google sent back to you
    code = request.args.get("code")

    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Prepare and send a request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Now that you have tokens (yay) let's find and hit the URL
    # from Google that gives you the user's profile information,
    # including their Google profile image and email
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    # You want to make sure their email is verified.
    # The user authenticated with Google, authorized your
    # app, and now you've verified their email through Google!
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400

    print(unique_id, "unique_id")
    user = User(
        id=str(unique_id),
        username=users_name,
        email=users_email
    )

    # Doesn't exist? Add it to the database.
    #TODO make password nullable
    if not load_user(user.id):
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            flash(f"Such an account has already been created not via Google!", 'danger')
            return redirect(url_for("login"))

    # Begin user session by logging the user in
    login_user(user)

    flash(f"Successfully logged into {users_name}!", 'success')

    # Send user back to homepage
    return redirect(url_for("home"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/result")
def result():
    return render_template('result.html', title="Result")

def generate_image(text, filename, image_url=None, denoising_strength=0.5):
    if image_url:
        image_path = "disruptor" + image_url
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        image_query({'prompt': text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed",
                     "sampler_name": "DPM2 Karras", "init_images": [input_image_b64],
                     "cfg_scale": 7.5,
                     "denoising_strength": denoising_strength,
                     "negative_prompt": "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish",
                     },
                    filename)
    else:
        text_query({
            'prompt': text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed",
            "sampler_name": "DPM2 Karras",
            "negative_prompt": "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish",
                    }, filename)
def generate_favourites(text, image_url=None, similar_image_number=2, different_image_number=4):
    for i in range(similar_image_number+different_image_number):
        # TODO check if DPM2 Karras parameter indeed works
        if image_url: # If we chose a favourite, we generate a new image from it
            if i > different_image_number - 1: # After we generated enough different images, we generate similar images
                generate_image(text, f'fav{i}.jpg', image_url, denoising_strength=0.2)
            else: # Generate significantly different images
                generate_image(text, f'fav{i}.jpg', image_url, denoising_strength=0.9)
        else:
            generate_image(text, f'fav{i}.jpg')

@app.route("/favourites")
def favourites():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    chosen_favourite = request.args.get('chosen_favourite')
    if chosen_favourite: # favourite page -> favourite page
        pass
        # Generate images from image + text
        # set_deliberate() # Set the correct model for better results
        # generate_image(text, "current_image.jpg", image_url)
        # Update the left-side image
        # image_url = url_for('static', filename="images/current_image.jpg")
        # generate_favourites(text, image_url)
    else:# If we chose go to favorites from style page
        pass
        #Generate option images from text
        # set_deliberate() # Set the correct model for better results
        # generate_favourites(text)
    return render_template('favourites.html', title="Favourites", text=text, image_url=image_url, chosen_favourite=chosen_favourite)

@app.route("/style")
def style():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    return render_template('style.html', title="Style", image_url=image_url, text=text)
@app.route("/budget")
def budget():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    return render_template('budget.html', title="Budget", image_url=image_url, text=text)

@app.route("/room")
def room():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    print(image_url, "IMAGE_URL")
    print(text, "TEXT")
    return render_template('room.html', title="Room", image_url=image_url, text=text)

@app.route("/space")
def space():
    return render_template('space.html', title="Space")

@app.route("/applyStyle")
def applyStyle():
    return render_template('applyStyle.html', title="applyStyle")

@app.route("/save_image", methods=["POST"])
def save_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to the specified folder on the server
        filename = file.filename
        print(url_for('static', filename=f'images/{filename}'))
        file_path = "disruptor/static/images/" + filename
        file.save(file_path)

        # Return the URL of the saved image
        return jsonify({'url': url_for('static', filename=f'images/{filename}')})
        # return jsonify({'url': file_path})

    return jsonify({'error': 'Unknown error'})