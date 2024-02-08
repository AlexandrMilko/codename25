from flask import redirect, render_template, url_for, flash, request, jsonify
from disruptor.forms import RegistrationForm, LoginForm, RequestPasswordResetForm, PasswordResetForm
from disruptor import app, db, mail
from flask_bcrypt import generate_password_hash, check_password_hash
from disruptor.models import User, load_user
from flask_login import (
    login_required,
    login_user,
    logout_user,
    current_user
)
from disruptor.google_auth import *
from disruptor.sdquery import TextQuery, ImageQuery, ControlNetImageQuery, apply_style
import base64
import json
import uuid
import os
from sqlalchemy.exc import IntegrityError
from flask_mail import Message

@app.route("/")
@app.route("/home")
def home():
    return render_template('homepage.html', title="Home")

@app.route("/register", methods=["GET", "POST"])
#TODO: remove to be able to register
# @login_required
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
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
        send_verification_email(user)
        flash("An email has been sent with instructions to verify your account.", "info")
        return redirect(url_for('login'))
    return render_template('register.html', title="Register", form=form)

def send_verification_email(user):
    token = user.get_email_verification_token()
    msg = Message(
        "Codename25: Email Verification",
                  sender='milkoaleksandr21@gmail.com',
                  recipients=[user.email]
                )
    msg.body = f'''To verify your account, visit the following link:
{url_for('verify_email', token=token, _external=True)}
If you did not make this request, then simply ignore this email and no changes will be made.
    '''
    mail.send(msg)
@app.route("/verify_email/<token>", methods=["GET", "POST"])
def verify_email(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user_data = User.verify_email_token(token)
    if user_data is None:
        flash('That is an expired or invalid token.', 'warning')
        return redirect(url_for('home'))
    else:
        user = User(
                id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                password=user_data["password"]
            )
        db.session.add(user)
        db.session.commit()
        flash(f"Your account has been verified!", 'success')
        return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
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
#TODO: remove to be able to register
# @login_required
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
@login_required
def result():
    text = request.args.get("text")
    return render_template('result.html', title="Result", text=text)

def generate_image(text, filename, image_url=None, denoising_strength=None):
    if image_url:
        query = ImageQuery(text, image_url, filename, denoising_strength)
        query.run()
    else:
        query = TextQuery(text, filename)
        query.run()

def generate_favourites(text, image_url=None, similar_image_number=2, different_image_number=4):
    for i in range(similar_image_number+different_image_number):
        # TODO check if DPM2 Karras parameter indeed works
        if image_url: # If we chose a favourite, we generate a new image from it
            if i > different_image_number - 1: # After we generated enough different images, we generate similar images
                generate_image(text, f'fav{i}.jpg', image_url, denoising_strength=0.55)
            else: # Generate significantly different images
                generate_image(text, f'fav{i}.jpg', image_url, denoising_strength=0.9)
        else:
            generate_image(text, f'fav{i}.jpg')

@app.route("/favourites")
@login_required
def favourites():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    chosen_favourite = request.args.get('chosen_favourite')
    if chosen_favourite: # favourite page -> favourite page
        # Generate images from image + text
        generate_image(text, "current_image.jpg", image_url, denoising_strength=0.5)
        # Update the left-side image
        image_url = url_for('static', filename=f"images/{current_user.id}/current_image.jpg")
        generate_favourites(text, image_url)
    else:# If we chose go to favorites from style page
        # Generate option images from text
        generate_favourites(text)
    image_paths = [f'images/{current_user.id}/fav{i}.jpg' for i in range(6)]
    return render_template('favourites.html', title="Favourites", text=text, image_url=image_url, chosen_favourite=chosen_favourite, image_paths=image_paths)

@app.route("/style")
@login_required
def style():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    return render_template('style.html', title="Style", image_url=image_url, text=text)
@app.route("/budget")
@login_required
def budget():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    return render_template('budget.html', title="Budget", image_url=image_url, text=text)

@app.route("/room")
@login_required
def room():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    return render_template('room.html', title="Room", image_url=image_url, text=text)

@app.route("/space")
@login_required
def space():
    return render_template('space.html', title="Space")

@app.route("/applyStyle")
@login_required
def applyStyle():
    text = request.args.get("text")
    return render_template('applyStyle.html', title="applyStyle", text=text)

@app.route("/save_image", methods=["POST"])
@login_required
def save_image():
    text = request.form['text']
    print(text, "THE TEXT")
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to the specified folder on the server
        filename = file.filename
        directory = f"disruptor/static/images/{current_user.id}/"
        from disruptor.tools import create_directory_if_not_exists
        create_directory_if_not_exists(directory)
        file_path = directory + filename
        file.save(file_path)

        apply_style(filename, text)

        # Return the URL of the saved image
        return jsonify({'url': url_for('static', filename=f'images/{current_user.id}/applied.jpg')})
        # return jsonify({'url': file_path})

    return jsonify({'error': 'Unknown error'})

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message(
        "Codename25: Password Reset Request",
                  sender='milkoaleksandr21@gmail.com',
                  recipients=[user.email]
                )
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_password', token=token, _external=True)}
If you did not make this request, then simply ignore this email and no changes will be made.
    '''
    mail.send(msg)

@app.route("/request_password_reset", methods=["GET", "POST"])
def request_password_reset():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestPasswordResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash("An email has been sent with instructions to reset your password.", "info")
        return redirect(url_for('login'))
    return render_template('request_password_reset.html', title="Reset Password", form=form)

@app.route("/request_password_reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an expired or invalid token.', 'warning')
        return redirect(url_for('request_password_reset'))
    form = PasswordResetForm()
    if form.validate_on_submit():
        password_hash = generate_password_hash(form.password.data).decode("utf-8")
        user.password = password_hash
        db.session.commit()
        flash(f"Password has been successfully updated!", 'success')
        return redirect(url_for('login'))
    return render_template('password_reset.html', title="Reset Password", form=form)

@app.route("/test", methods=["GET"])
def test():
    from disruptor.staging_ml import Room
    df = Room.get_paired_dataframe(r"C:\Users\Sasha\Desktop\Projects\codename25\src\disruptor\green_screen\find_similar\dataset\bedroom\es_segmented")
    return redirect(url_for('home'))