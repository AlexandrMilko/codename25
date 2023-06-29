from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import Length, DataRequired, Email, EqualTo, ValidationError
from disruptor.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = StringField('Password',
                           validators=[DataRequired()])
    confirm_password = StringField('Confirm Password',
                                   validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError("Such username is already taken. Choose another one.")

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError("Such email is already taken. Choose another one.")

class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = StringField('Password',
                           validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Sign In')