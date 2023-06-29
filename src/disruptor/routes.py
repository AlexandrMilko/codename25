from flask import redirect, render_template, url_for, flash, request
from disruptor.forms import RegistrationForm, LoginForm
from disruptor import app, db
from flask_bcrypt import generate_password_hash, check_password_hash
from disruptor.models import User
from flask_login import login_user, logout_user, login_required

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