from flask import Flask, render_template, flash, redirect, url_for
from forms import RegistrationForm, LoginForm
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SECRET_KEY"] = "8a9060645436c83806084ef6aa20547c"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"User({self.id}, {self.username}, {self.email})"

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

if __name__ == "__main__":
    app.run(debug=True)