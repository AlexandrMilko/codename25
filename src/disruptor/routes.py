from flask import redirect, render_template, url_for, flash
from disruptor.forms import RegistrationForm, LoginForm
from disruptor import app

@app.route("/")
@app.route("/home")
def home():
    return render_template('homepage.html', title="Home")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f"Account created for {form.username.data}!", 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title="Register", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == "hello@gmail.com" and form.password.data == "123123":
            flash(f"Successfully logged in", 'success')
            return redirect(url_for("home"))
        else:
            flash(f"Wrong email or password", 'danger')
            return redirect(url_for('login'))
    return render_template('login.html', title="Login", form=form)