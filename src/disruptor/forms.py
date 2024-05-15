from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import Length, DataRequired, Email, EqualTo, ValidationError
from disruptor.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username', # TODO Have no idea why but it does not change the message to English
                           validators=[DataRequired(message="Please, fill in this field"), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(message="Please, fill in this field"), Email()])
    password = StringField('Password',
                           validators=[DataRequired(message="Please, fill in this field")])
    confirm_password = StringField('Confirm Password',
                                   validators=[DataRequired(message="Please, fill in this field"), EqualTo('password')])
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

class RequestPasswordResetForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(message="Please, fill in this field"), Email()])
    submit = SubmitField('Request Password Reset')
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError("No account with that email.")

class PasswordResetForm(FlaskForm):
    password = StringField('Password',
                           validators=[DataRequired(message="Please, fill in this field")])
    confirm_password = StringField('Confirm Password',
                                   validators=[DataRequired(message="Please, fill in this field"), EqualTo('password')])
    submit = SubmitField('Reset Password')