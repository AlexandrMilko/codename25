from flask import redirect, render_template, url_for, flash, request
from disruptor.forms import RegistrationForm, LoginForm
from disruptor import app, db
from flask_bcrypt import generate_password_hash, check_password_hash
from disruptor.models import User
from flask_login import login_user, logout_user, login_required
from disruptor.sdquery import text_query, image_query
import base64
import os

@app.route("/")
@app.route("/home")
def home():
    return render_template('homepage.html', title="Home")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        password_hash = generate_password_hash(form.password.data).decode("utf-8")
        flash(f"Account created for {form.username.data}!", 'success')
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=password_hash
                    )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', title="Register", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            flash(f"Successfully logged into {user.username}", 'success')
            next_page = request.args.get('next')
            return redirect(url_for(next_page)) if next_page else redirect(url_for('home'))
        else:
            flash(f"Wrong email or password", 'danger')
    return render_template('login.html', title="Login", form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/result")
def result():
    return render_template('result.html', title="Result")

def generate_image(text, filename, image_url=None):
    if image_url:
        image_path = "disruptor" + image_url
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        image_query({'prompt': text + ", interior design, 4k, ultra-realistic",
                     "sampler_name": "DPM2", "init_images": [input_image_b64],
                     "cfg_scale": 7.5,
                     "denoising_strength": 0.5,
                     },
                    filename)
    else:
        text_query({'prompt': text + ", interior design, 4k, ultra-realistic",
                    "sampler_name": "DPM2"}, filename)
def generate_favourites(text, image_url=None, image_number=4):
    for i in range(image_number):
        # TODO check if DPM2 Karras parameter indeed works
        if image_url:
            generate_image(text, f'fav{i}.jpg', image_url)
        else:
            generate_image(text, f'fav{i}.jpg')
@app.route("/favourites")
def favourites():
    image_url = request.args.get("image_url")
    text = request.args.get("text")
    chosen_favourite = request.args.get('chosen_favourite')
    if chosen_favourite: # favourite page -> favourite page
        # Generate images from image + text
        generate_image(text, "current_image.jpg", image_url)
        # Update the left-side image
        image_url = url_for('static', filename="images/current_image.jpg")
        generate_favourites(text, image_url)
    else:# If we chose go to favorites from style page
        #Generate option images from text
        generate_favourites(text)
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